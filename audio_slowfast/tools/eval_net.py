from loguru import logger
import torch
from tqdm import tqdm
import wandb
from audio_slowfast.utils import metrics
import torch
import wandb
import audio_slowfast.models.losses as losses
import audio_slowfast.utils.distributed as du
import audio_slowfast.utils.metrics as metrics
import audio_slowfast.tools.train_utils as train_utils
from src.dataset import load_all_verbs, load_nouns


@torch.no_grad()
def eval_epoch_with_state(
    val_loader,
    model,
    val_meter,
    cur_epoch,
    cfg,
    writer=None,
    wandb_log=False,
):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    val_meter.iter_tic()

    verb_names = load_all_verbs(cfg.EPICKITCHENS.VERBS_FILE)
    noun_names = load_nouns(cfg.EPICKITCHENS.NOUNS_FILE)

    for cur_iter, (inputs, lengths, labels, _, noun_embeddings, _) in enumerate(
        tqdm(
            val_loader,
            desc="Validation at epoch {}".format(
                cur_epoch,
            ),
            unit="batch",
        ),
    ):
        # Transfer the data to the current GPU device.
        if cfg.NUM_GPUS:
            # Transferthe data to the current GPU device.
            if isinstance(inputs, list):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            if isinstance(labels, dict):
                labels = {k: v.cuda() for k, v in labels.items()}
            else:
                labels = labels.cuda()

            noun_embeddings = noun_embeddings.cuda()

        val_meter.data_toc()
        logger.warning([i.shape for i in inputs])
        preds = model(
            x=inputs,
            lengths=lengths,
            noun_embeddings=noun_embeddings,
        )

        verb_preds, noun_preds, state_preds = preds

        labels["state"] = train_utils.prepare_state_labels(preds, labels, lengths)

        if isinstance(labels, dict):
            loss, loss_verb, loss_noun, loss_state = train_utils.compute_loss_with_state(
                verb_preds=verb_preds,
                noun_preds=noun_preds,
                state_preds=state_preds,
                labels=labels,
                cfg=cfg,
            )

            # Compute the verb accuracies.
            verb_top1_acc, verb_top5_acc = metrics.topk_accuracies(verb_preds, labels["verb"], (1, 5))

            # Combine the errors across the GPUs.
            if cfg.NUM_GPUS > 1:
                loss_verb, verb_top1_acc, verb_top5_acc = du.all_reduce([loss_verb, verb_top1_acc, verb_top5_acc])

            # Copy the errors from GPU to CPU (sync point).
            loss_verb, verb_top1_acc, verb_top5_acc = (
                loss_verb.item(),
                verb_top1_acc.item(),
                verb_top5_acc.item(),
            )

            # Compute the noun accuracies.
            noun_top1_acc, noun_top5_acc = metrics.topk_accuracies(noun_preds, labels["noun"], (1, 5))

            # Combine the errors across the GPUs.
            if cfg.NUM_GPUS > 1:
                loss_noun, noun_top1_acc, noun_top5_acc = du.all_reduce([loss_noun, noun_top1_acc, noun_top5_acc])

            # Copy the errors from GPU to CPU (sync point).
            loss_noun, noun_top1_acc, noun_top5_acc = (
                loss_noun.item(),
                noun_top1_acc.item(),
                noun_top5_acc.item(),
            )

            # Compute the action accuracies.
            action_top1_acc, action_top5_acc = metrics.multitask_topk_accuracies(
                (verb_preds, noun_preds),
                (labels["verb"], labels["noun"]),
                (1, 5),
            )
            # Combine the errors across the GPUs.
            if cfg.NUM_GPUS > 1:
                loss, action_top1_acc, action_top5_acc = du.all_reduce([loss, action_top1_acc, action_top5_acc])

            # Copy the errors from GPU to CPU (sync point).
            loss, action_top1_acc, action_top5_acc = (
                loss.item(),
                action_top1_acc.item(),
                action_top5_acc.item(),
            )

            val_meter.iter_toc()
            # Update and log stats.
            val_meter.update_stats(
                top1_acc=(
                    verb_top1_acc,
                    noun_top1_acc,
                    action_top1_acc,
                ),
                top5_acc=(
                    verb_top5_acc,
                    noun_top5_acc,
                    action_top5_acc,
                ),
                mb_size=inputs[0].size(0)
                * max(cfg.NUM_GPUS, 1),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
            )
            # write to tensorboard format if available.
            if writer is not None and not wandb_log:
                writer.add_scalars(
                    {
                        "Val/loss": loss,
                        "Val/Top1_acc": action_top1_acc,
                        "Val/Top5_acc": action_top5_acc,
                        "Val/verb/loss": loss_verb,
                        "Val/verb/Top1_acc": verb_top1_acc,
                        "Val/verb/Top5_acc": verb_top5_acc,
                        "Val/noun/loss": loss_noun,
                        "Val/noun/Top1_acc": noun_top1_acc,
                        "Val/noun/Top5_acc": noun_top5_acc,
                        "Val/state/loss": loss_state,
                    },
                    global_step=len(val_loader) * cur_epoch + cur_iter,
                )

            if wandb_log:
                # Log confusion matrix for verb and noun
                verb_confusion_matrix = wandb.plot.confusion_matrix(
                    probs=verb_preds.detach().cpu().numpy(),
                    y_true=labels["verb"].detach().cpu().numpy(),
                    class_names=verb_names,
                )
                noun_confusion_matrix = wandb.plot.confusion_matrix(
                    probs=noun_preds.detach().cpu().numpy(),
                    y_true=labels["noun"].detach().cpu().numpy(),
                    class_names=noun_names,
                )
                wandb.log(
                    {
                        "Val/loss": loss,
                        "Val/Top1_acc": action_top1_acc,
                        "Val/Top5_acc": action_top5_acc,
                        "Val/verb/loss": loss_verb,
                        "Val/verb/Top1_acc": verb_top1_acc,
                        "Val/verb/Top5_acc": verb_top5_acc,
                        "Val/noun/loss": loss_noun,
                        "Val/noun/Top1_acc": noun_top1_acc,
                        "Val/noun/Top5_acc": noun_top5_acc,
                        "Val/state/loss": loss_state,
                        "val_step": len(val_loader) * cur_epoch + cur_iter,
                    },
                )

            val_meter.update_predictions(
                preds=(
                    verb_preds,
                    noun_preds,
                    state_preds,
                ),
                labels=(
                    labels["verb"],
                    labels["noun"],
                    labels["state"],
                ),
            )

        else:
            # Explicitly declare reduction to mean.
            loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")

            # Compute the loss.
            loss = loss_fun(preds, labels)

            if cfg.DATA.MULTI_LABEL:
                if cfg.NUM_GPUS > 1:
                    preds, labels = du.all_gather([preds, labels])

            else:
                # Compute the errors.
                num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))

                # Combine the errors across the GPUs.
                top1_err, top5_err = [(1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct]
                if cfg.NUM_GPUS > 1:
                    loss, top1_err, top5_err = du.all_reduce([loss, top1_err, top5_err])

                # Copy the errors from GPU to CPU (sync point).
                loss, top1_err, top5_err = (
                    loss.item(),
                    top1_err.item(),
                    top5_err.item(),
                )

                val_meter.iter_toc()
                # Update and log stats.
                val_meter.update_stats(
                    top1_err,
                    top5_err,
                    inputs[0].size(0)
                    * max(cfg.NUM_GPUS, 1),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
                )
                # write to tensorboard format if available.
                if writer is not None and not wandb_log:
                    writer.add_scalars(
                        {
                            "Val/loss": loss,
                            "Val/Top1_err": top1_err,
                            "Val/Top5_err": top5_err,
                        },
                        global_step=len(val_loader) * cur_epoch + cur_iter,
                    )

                if wandb_log:
                    wandb.log(
                        {
                            "Val/loss": loss,
                            "Val/Top1_err": top1_err,
                            "Val/Top5_err": top5_err,
                            "val_step": len(val_loader) * cur_epoch + cur_iter,
                        },
                    )

            val_meter.update_predictions(preds, labels)

        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

    # Log epoch stats.
    is_best_epoch, top1_dict = val_meter.log_epoch_stats(cur_epoch)
    # write to tensorboard format if available.
    if writer is not None:
        all_verb_preds = [pred.clone().detach() for pred in val_meter.all_verb_preds]
        all_verb_labels = [label.clone().detach() for label in val_meter.all_verb_labels]
        all_noun_preds = [pred.clone().detach() for pred in val_meter.all_noun_preds]
        all_noun_labels = [label.clone().detach() for label in val_meter.all_noun_labels]
        all_state_preds = [pred.clone().detach() for pred in val_meter.all_state_preds]
        all_state_labels = [label.clone().detach() for label in val_meter.all_state_labels]

        if cfg.NUM_GPUS:
            all_verb_preds = [pred.cpu() for pred in all_verb_preds]
            all_verb_labels = [label.cpu() for label in all_verb_labels]
            all_noun_preds = [pred.cpu() for pred in all_noun_preds]
            all_noun_labels = [label.cpu() for label in all_noun_labels]
            all_state_preds = [pred.cpu() for pred in all_state_preds]
            all_state_labels = [label.cpu() for label in all_state_labels]

        writer.plot_eval(preds=all_verb_preds, labels=all_verb_labels, global_step=cur_epoch)
        writer.plot_eval(preds=all_noun_preds, labels=all_noun_labels, global_step=cur_epoch)
        writer.plot_eval(preds=all_state_preds, labels=all_state_labels, global_step=cur_epoch)

    if writer is not None and not wandb_log:
        if "top1_acc" in top1_dict.keys():
            writer.add_scalars(
                {
                    "Val/epoch/Top1_acc": top1_dict["top1_acc"],
                    "Val/epoch/verb/Top1_acc": top1_dict["verb_top1_acc"],
                    "Val/epoch/noun/Top1_acc": top1_dict["noun_top1_acc"],
                },
                global_step=cur_epoch,
            )

        else:
            writer.add_scalars(
                {"Val/epoch/Top1_err": top1_dict["top1_err"]},
                global_step=cur_epoch,
            )

    if wandb_log:
        if "top1_acc" in top1_dict.keys():
            wandb.log(
                {
                    "Val/epoch/Top1_acc": top1_dict["top1_acc"],
                    "Val/epoch/verb/Top1_acc": top1_dict["verb_top1_acc"],
                    "Val/epoch/noun/Top1_acc": top1_dict["noun_top1_acc"],
                    "epoch": cur_epoch,
                },
            )

        else:
            wandb.log({"Val/epoch/Top1_err": top1_dict["top1_err"], "epoch": cur_epoch})

    top1 = top1_dict["top1_acc"] if "top1_acc" in top1_dict.keys() else top1_dict["top1_err"]
    val_meter.reset()
    return is_best_epoch, top1


@torch.no_grad()
def eval_epoch(
    val_loader,
    model,
    val_meter,
    cur_epoch,
    cfg,
    writer=None,
    wandb_log=False,
):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    val_meter.iter_tic()

    verb_names = load_all_verbs(cfg.EPICKITCHENS.VERBS_FILE)
    noun_names = load_nouns(cfg.EPICKITCHENS.NOUNS_FILE)

    for cur_iter, (inputs, lengths, labels, _, noun_embeddings, _) in enumerate(
        tqdm(
            val_loader,
            desc="Validation at epoch {}".format(
                cur_epoch,
            ),
            unit="batch",
        ),
    ):
        # Transfer the data to the current GPU device.
        if cfg.NUM_GPUS:
            # Transferthe data to the current GPU device.
            if isinstance(inputs, list):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            if isinstance(labels, dict):
                labels = {k: v.cuda() for k, v in labels.items()}
            else:
                labels = labels.cuda()

            noun_embeddings = noun_embeddings.cuda()

        val_meter.data_toc()
        logger.warning([i.shape for i in inputs])
        preds = model(
            x=inputs,
            lengths=lengths,
            noun_embeddings=noun_embeddings,
        )

        verb_preds, noun_preds = preds

        if isinstance(labels, dict):
            loss, loss_verb, loss_noun = train_utils.compute_loss(
                verb_preds=verb_preds,
                noun_preds=noun_preds,
                labels=labels,
                cfg=cfg,
            )

            # Compute the verb accuracies.
            verb_top1_acc, verb_top5_acc = metrics.topk_accuracies(verb_preds, labels["verb"], (1, 5))

            # Combine the errors across the GPUs.
            if cfg.NUM_GPUS > 1:
                loss_verb, verb_top1_acc, verb_top5_acc = du.all_reduce([loss_verb, verb_top1_acc, verb_top5_acc])

            # Copy the errors from GPU to CPU (sync point).
            loss_verb, verb_top1_acc, verb_top5_acc = (
                loss_verb.item(),
                verb_top1_acc.item(),
                verb_top5_acc.item(),
            )

            # Compute the noun accuracies.
            noun_top1_acc, noun_top5_acc = metrics.topk_accuracies(noun_preds, labels["noun"], (1, 5))

            # Combine the errors across the GPUs.
            if cfg.NUM_GPUS > 1:
                loss_noun, noun_top1_acc, noun_top5_acc = du.all_reduce([loss_noun, noun_top1_acc, noun_top5_acc])

            # Copy the errors from GPU to CPU (sync point).
            loss_noun, noun_top1_acc, noun_top5_acc = (
                loss_noun.item(),
                noun_top1_acc.item(),
                noun_top5_acc.item(),
            )

            # Compute the action accuracies.
            action_top1_acc, action_top5_acc = metrics.multitask_topk_accuracies(
                (verb_preds, noun_preds),
                (labels["verb"], labels["noun"]),
                (1, 5),
            )
            # Combine the errors across the GPUs.
            if cfg.NUM_GPUS > 1:
                loss, action_top1_acc, action_top5_acc = du.all_reduce([loss, action_top1_acc, action_top5_acc])

            # Copy the errors from GPU to CPU (sync point).
            loss, action_top1_acc, action_top5_acc = (
                loss.item(),
                action_top1_acc.item(),
                action_top5_acc.item(),
            )

            val_meter.iter_toc()
            # Update and log stats.
            val_meter.update_stats(
                top1_acc=(
                    verb_top1_acc,
                    noun_top1_acc,
                    action_top1_acc,
                ),
                top5_acc=(
                    verb_top5_acc,
                    noun_top5_acc,
                    action_top5_acc,
                ),
                mb_size=inputs[0].size(0)
                * max(cfg.NUM_GPUS, 1),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
            )
            # write to tensorboard format if available.
            if writer is not None and not wandb_log:
                writer.add_scalars(
                    {
                        "Val/loss": loss,
                        "Val/Top1_acc": action_top1_acc,
                        "Val/Top5_acc": action_top5_acc,
                        "Val/verb/loss": loss_verb,
                        "Val/verb/Top1_acc": verb_top1_acc,
                        "Val/verb/Top5_acc": verb_top5_acc,
                        "Val/noun/loss": loss_noun,
                        "Val/noun/Top1_acc": noun_top1_acc,
                        "Val/noun/Top5_acc": noun_top5_acc,
                    },
                    global_step=len(val_loader) * cur_epoch + cur_iter,
                )

            if wandb_log:
                # Log confusion matrix for verb and noun
                verb_confusion_matrix = wandb.plot.confusion_matrix(
                    probs=verb_preds.detach().cpu().numpy(),
                    y_true=labels["verb"].detach().cpu().numpy(),
                    class_names=verb_names,
                )
                noun_confusion_matrix = wandb.plot.confusion_matrix(
                    probs=noun_preds.detach().cpu().numpy(),
                    y_true=labels["noun"].detach().cpu().numpy(),
                    class_names=noun_names,
                )
                wandb.log(
                    {
                        "Val/loss": loss,
                        "Val/Top1_acc": action_top1_acc,
                        "Val/Top5_acc": action_top5_acc,
                        "Val/verb/loss": loss_verb,
                        "Val/verb/Top1_acc": verb_top1_acc,
                        "Val/verb/Top5_acc": verb_top5_acc,
                        "Val/noun/loss": loss_noun,
                        "Val/noun/Top1_acc": noun_top1_acc,
                        "Val/noun/Top5_acc": noun_top5_acc,
                        "val_step": len(val_loader) * cur_epoch + cur_iter,
                    },
                )

            val_meter.update_predictions(
                preds=(
                    verb_preds,
                    noun_preds,
                ),
                labels=(
                    labels["verb"],
                    labels["noun"],
                ),
            )

        else:
            # Explicitly declare reduction to mean.
            loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")

            # Compute the loss.
            loss = loss_fun(preds, labels)

            if cfg.DATA.MULTI_LABEL:
                if cfg.NUM_GPUS > 1:
                    preds, labels = du.all_gather([preds, labels])

            else:
                # Compute the errors.
                num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))

                # Combine the errors across the GPUs.
                top1_err, top5_err = [(1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct]
                if cfg.NUM_GPUS > 1:
                    loss, top1_err, top5_err = du.all_reduce([loss, top1_err, top5_err])

                # Copy the errors from GPU to CPU (sync point).
                loss, top1_err, top5_err = (
                    loss.item(),
                    top1_err.item(),
                    top5_err.item(),
                )

                val_meter.iter_toc()
                # Update and log stats.
                val_meter.update_stats(
                    top1_err,
                    top5_err,
                    inputs[0].size(0)
                    * max(cfg.NUM_GPUS, 1),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
                )
                # write to tensorboard format if available.
                if writer is not None and not wandb_log:
                    writer.add_scalars(
                        {
                            "Val/loss": loss,
                            "Val/Top1_err": top1_err,
                            "Val/Top5_err": top5_err,
                        },
                        global_step=len(val_loader) * cur_epoch + cur_iter,
                    )

                if wandb_log:
                    wandb.log(
                        {
                            "Val/loss": loss,
                            "Val/Top1_err": top1_err,
                            "Val/Top5_err": top5_err,
                            "val_step": len(val_loader) * cur_epoch + cur_iter,
                        },
                    )

            val_meter.update_predictions(preds, labels)

        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

    # Log epoch stats.
    is_best_epoch, top1_dict = val_meter.log_epoch_stats(cur_epoch)
    # write to tensorboard format if available.
    if writer is not None:
        all_verb_preds = [pred.clone().detach() for pred in val_meter.all_verb_preds]
        all_verb_labels = [label.clone().detach() for label in val_meter.all_verb_labels]
        all_noun_preds = [pred.clone().detach() for pred in val_meter.all_noun_preds]
        all_noun_labels = [label.clone().detach() for label in val_meter.all_noun_labels]

        if cfg.NUM_GPUS:
            all_verb_preds = [pred.cpu() for pred in all_verb_preds]
            all_verb_labels = [label.cpu() for label in all_verb_labels]
            all_noun_preds = [pred.cpu() for pred in all_noun_preds]
            all_noun_labels = [label.cpu() for label in all_noun_labels]

        writer.plot_eval(preds=all_verb_preds, labels=all_verb_labels, global_step=cur_epoch)
        writer.plot_eval(preds=all_noun_preds, labels=all_noun_labels, global_step=cur_epoch)

    if writer is not None and not wandb_log:
        if "top1_acc" in top1_dict.keys():
            writer.add_scalars(
                {
                    "Val/epoch/Top1_acc": top1_dict["top1_acc"],
                    "Val/epoch/verb/Top1_acc": top1_dict["verb_top1_acc"],
                    "Val/epoch/noun/Top1_acc": top1_dict["noun_top1_acc"],
                },
                global_step=cur_epoch,
            )

        else:
            writer.add_scalars(
                {"Val/epoch/Top1_err": top1_dict["top1_err"]},
                global_step=cur_epoch,
            )

    if wandb_log:
        if "top1_acc" in top1_dict.keys():
            wandb.log(
                {
                    "Val/epoch/Top1_acc": top1_dict["top1_acc"],
                    "Val/epoch/verb/Top1_acc": top1_dict["verb_top1_acc"],
                    "Val/epoch/noun/Top1_acc": top1_dict["noun_top1_acc"],
                    "epoch": cur_epoch,
                },
            )

        else:
            wandb.log({"Val/epoch/Top1_err": top1_dict["top1_err"], "epoch": cur_epoch})

    top1 = top1_dict["top1_acc"] if "top1_acc" in top1_dict.keys() else top1_dict["top1_err"]
    val_meter.reset()
    return is_best_epoch, top1
