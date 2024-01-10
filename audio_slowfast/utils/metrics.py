#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Functions for computing metrics."""

from loguru import logger
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score
import torch
import torch.masked.maskedtensor


def topks_correct(preds, labels, ks):
    """
    Given the predictions, labels, and a list of top-k values, compute the
    number of correct predictions for each top-k value.

    Args:
        preds (array): array of predictions. Dimension is batchsize
            N x ClassNum.
        labels (array): array of labels. Dimension is batchsize N.
        ks (list): list of top-k values. For example, ks = [1, 5] correspods
            to top-1 and top-5.

    Returns:
        topks_correct (list): list of numbers, where the `i`-th entry
            corresponds to the number of top-`ks[i]` correct predictions.
    """
    assert preds.size(0) == labels.size(0), "Batch dim of predictions and labels must match"
    # Find the top max_k predictions for each sample
    _top_max_k_vals, top_max_k_inds = torch.topk(preds, max(ks), dim=1, largest=True, sorted=True)
    # (batch_size, max_k) -> (max_k, batch_size).
    top_max_k_inds = top_max_k_inds.t()
    # (batch_size, ) -> (max_k, batch_size).
    rep_max_k_labels = labels.view(1, -1).expand_as(top_max_k_inds)
    # (i, j) = 1 if top i-th prediction for the j-th sample is correct.
    top_max_k_correct = top_max_k_inds.eq(rep_max_k_labels)
    # Compute the number of topk correct predictions for each k.
    topks_correct = [top_max_k_correct[:k, :].float().sum() for k in ks]
    return topks_correct


def topks_correct_slide(preds, labels, ks, per_action_instance=True, weight=None):
    """
    Given the predictions, labels, and a list of top-k values, compute the
    number of correct predictions for each top-k value.

    Args:
        preds (array): array of predictions. Dimension is batchsize
            N x ClassNum.
        labels (array): array of labels. Dimension is batchsize N.
        ks (list): list of top-k values. For example, ks = [1, 5] correspods
            to top-1 and top-5.

    Returns:
        topks_correct (list): list of numbers, where the `i`-th entry
            corresponds to the number of top-`ks[i]` correct predictions.
    """
    weight = torch.ones(preds.size(0)) / preds.size(0) if weight == None else weight / torch.sum(weight)
    assert preds.size(0) == labels.size(0), "Batch dim of predictions and labels must match"
    # Find the top max_k predictions for each sample
    _top_max_k_vals, top_max_k_inds = torch.topk(preds, max(ks), dim=1, largest=True, sorted=True)
    # (batch_size, max_k) -> (max_k, batch_size).
    top_max_k_inds = top_max_k_inds.t()
    # (batch_size, ) -> (max_k, batch_size).
    if per_action_instance:
        rep_max_k_labels = labels.view(1, -1).expand_as(top_max_k_inds)
        top_max_k_correct = top_max_k_inds.eq(rep_max_k_labels)
    else:
        top_max_k_correct = torch.zeros_like(top_max_k_inds)
        for label in labels.t():
            rep_max_k_labels = label.view(1, -1).expand_as(top_max_k_inds)
            top_max_k_correct_ = top_max_k_inds.eq(rep_max_k_labels)
            top_max_k_correct |= top_max_k_correct_
    # (i, j) = 1 if top i-th prediction for the j-th sample is correct.
    # Compute the number of topk correct predictions for each k.
    topks_correct = [(weight * top_max_k_correct[:k, :]).contiguous().view(-1).float().sum() for k in ks]
    return topks_correct


def multitask_topks_correct(preds, labels, ks=(1,)):
    """
    Args:
        preds: tuple(torch.FloatTensor), each tensor should be of shape
            [batch_size, class_count], class_count can vary on a per task basis, i.e.
            outputs[i].shape[1] can be different to outputs[j].shape[j].
        labels: tuple(torch.LongTensor), each tensor should be of shape [batch_size]
        ks: tuple(int), compute accuracy at top-k for the values of k specified
            in this parameter.
    Returns:
        tuple(float), same length at topk with the corresponding accuracy@k in.
    """
    max_k = int(np.max(ks))
    task_count = len(preds)
    batch_size = labels[0].size(0)
    all_correct = torch.zeros(max_k, batch_size).type(torch.ByteTensor)
    if torch.cuda.is_available():
        all_correct = all_correct.cuda()
    for output, label in zip(preds, labels):
        _, max_k_idx = output.topk(max_k, dim=1, largest=True, sorted=True)
        # Flip batch_size, class_count as .view doesn't work on non-contiguous
        max_k_idx = max_k_idx.t()
        correct_for_task = max_k_idx.eq(label.view(1, -1).expand_as(max_k_idx))
        all_correct.add_(correct_for_task)

    multitask_topks_correct = [torch.ge(all_correct[:k].float().sum(0), task_count).float().sum(0) for k in ks]

    return multitask_topks_correct


def multitask_topks_correct_slide(preds, labels, ks=(1,), per_action_instance=True, weight=None):
    """
    Args:
        preds: tuple(torch.FloatTensor), each tensor should be of shape
            [batch_size, class_count], class_count can vary on a per task basis, i.e.
            outputs[i].shape[1] can be different to outputs[j].shape[j].
        labels: tuple(torch.LongTensor), each tensor should be of shape [batch_size]
        ks: tuple(int), compute accuracy at top-k for the values of k specified
            in this parameter.
    Returns:
        tuple(float), same length at topk with the corresponding accuracy@k in.
    """
    weight = torch.ones(preds[0].size(0)).to(preds[0].device) if weight is None else weight
    num_vids = torch.sum(weight)
    weight = weight / num_vids
    max_k = int(np.max(ks))
    task_count = len(preds)
    batch_size = labels[0].size(0)
    all_correct = torch.zeros(max_k, batch_size).type(torch.ByteTensor).to(preds[0].device)
    logger.debug(f"{all_correct.device=}")
    # if torch.cuda.is_available():
    #    all_correct = all_correct.cuda()
    for output, label in zip(preds, labels):
        _, max_k_idx = output.topk(max_k, dim=1, largest=True, sorted=True)
        # Flip batch_size, class_count as .view doesn't work on non-contiguous
        max_k_idx = max_k_idx.t()
        if per_action_instance:
            correct_for_task = max_k_idx.eq(label.view(1, -1).expand_as(max_k_idx))
        else:
            correct_for_task = torch.zeros_like(max_k_idx)
            for l in label.t():
                correct_for_task_ = max_k_idx.eq(l.view(1, -1).expand_as(max_k_idx))
                correct_for_task |= correct_for_task_
        logger.debug(f"{correct_for_task.device=}")
        all_correct.add_(correct_for_task)
    multitask_topks_correct = [
        (weight * torch.ge((all_correct[:k]).float().sum(0), task_count)).float().sum(0) for k in ks
    ]

    return multitask_topks_correct


def topk_errors(preds, labels, ks):
    """
    Computes the top-k error for each k.
    Args:
        preds (array): array of predictions. Dimension is N.
        labels (array): array of labels. Dimension is N.
        ks (list): list of ks to calculate the top accuracies.
    """
    num_topks_correct = topks_correct(preds, labels, ks)
    return [(1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct]


def topk_accuracies(preds, labels, ks):
    """
    Computes the top-k accuracy for each k.
    Args:
        preds (array): array of predictions. Dimension is N.
        labels (array): array of labels. Dimension is N.
        ks (list): list of ks to calculate the top accuracies.
    """
    num_topks_correct = topks_correct(preds, labels, ks)
    return [(x / preds.size(0)) * 100.0 for x in num_topks_correct]


def topk_accuracies_slide(preds, labels, ks, per_action_instance=True, weight=None):
    """
    Computes the top-k accuracy for each k.
    Args:
        preds (array): array of predictions. Dimension is N.
        labels (array): array of labels. Dimension is N.
        ks (list): list of ks to calculate the top accuracies.
    """
    num_topks_correct = topks_correct_slide(preds, labels, ks, per_action_instance, weight)
    return [x * 100.0 for x in num_topks_correct]


def multitask_topk_accuracies(preds, labels, ks):
    """
    Computes the top-k accuracy for each k.
    Args:
        preds (array): array of predictions. Dimension is N.
        labels (array): array of labels. Dimension is N.
        ks (list): list of ks to calculate the top accuracies.
    """
    num_multitask_topks_correct = multitask_topks_correct(preds, labels, ks)
    return [(x / preds[0].size(0)) * 100.0 for x in num_multitask_topks_correct]


def multitask_topk_accuracies_slide(preds, labels, ks, per_action_instance=True, weight=None):
    """
    Computes the top-k accuracy for each k.
    Args:
        preds (array): array of predictions. Dimension is N.
        labels (array): array of labels. Dimension is N.
        ks (list): list of ks to calculate the top accuracies.
    """
    num_multitask_topks_correct = multitask_topks_correct_slide(preds, labels, ks, per_action_instance, weight)
    return [x * 100.0 for x in num_multitask_topks_correct]


def state_metrics(preds, labels, lengths, split="Val"):
    """
    Computes the f1 score.
    Args:
        preds (array): array of predictions. Dimension is N.
        labels (array): array of labels. Dimension is N.
        lengths (array): array of lengths. Dimension is N.
    """
    softmax = torch.nn.Softmax(dim=3)
    if len(preds.shape) == 4:
        preds = softmax(preds)  # (B,N,P,3) -> (B,N,P,3)
        preds = preds.argmax(dim=3)  # (B,N,P,3) -> (B,N,P)
        labels = labels.argmax(dim=3)  # Remove the one-hot encoding
    else:
        preds = preds.mean(dim=2)  # (B,P,3) -> (B,P)
        labels = labels.argmax(dim=2)  # Remove the one-hot encoding

    B, N, P = preds.shape

    f1_macro_precs = torch.zeros(B, N)
    f1_micro_precs = torch.zeros(B, N)
    recall_macro_precs = torch.zeros(B, N)
    recall_micro_precs = torch.zeros(B, N)
    precision_macro_precs = torch.zeros(B, N)
    precision_micro_precs = torch.zeros(B, N)
    accuracy_precs = torch.zeros(B, N)

    f1_macro_posts = torch.zeros(B, N)
    f1_micro_posts = torch.zeros(B, N)
    recall_macro_posts = torch.zeros(B, N)
    recall_micro_posts = torch.zeros(B, N)
    precision_macro_posts = torch.zeros(B, N)
    precision_micro_posts = torch.zeros(B, N)
    accuracy_posts = torch.zeros(B, N)

    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()

    for i, length in enumerate(lengths):
        curr_pred_precs = preds[i, 0, :]
        curr_label_precs = labels[i, 0, :]

        f1_macro_precs[i, :] = f1_score(curr_label_precs, curr_pred_precs, average="macro", zero_division=0)
        f1_micro_precs[i, :] = f1_score(curr_label_precs, curr_pred_precs, average="micro", zero_division=0)
        recall_macro_precs[i, :] = recall_score(curr_label_precs, curr_pred_precs, average="macro", zero_division=0)
        recall_micro_precs[i, :] = recall_score(curr_label_precs, curr_pred_precs, average="micro", zero_division=0)
        precision_macro_precs[i, :] = precision_score(
            curr_label_precs, curr_pred_precs, average="macro", zero_division=0
        )
        precision_micro_precs[i, :] = precision_score(
            curr_label_precs, curr_pred_precs, average="micro", zero_division=0
        )
        accuracy_precs[i, :] = np.mean(curr_label_precs == curr_pred_precs)

        curr_pred_precs = preds[i, length - 1, :]
        curr_label_precs = labels[i, length - 1, :]

        f1_macro_posts[i, :] = f1_score(curr_label_precs, curr_pred_precs, average="macro", zero_division=0)
        f1_micro_posts[i, :] = f1_score(curr_label_precs, curr_pred_precs, average="micro", zero_division=0)
        recall_macro_posts[i, :] = recall_score(curr_label_precs, curr_pred_precs, average="macro", zero_division=0)
        recall_micro_posts[i, :] = recall_score(curr_label_precs, curr_pred_precs, average="micro", zero_division=0)
        precision_macro_posts[i, :] = precision_score(
            curr_label_precs, curr_pred_precs, average="macro", zero_division=0
        )
        precision_micro_posts[i, :] = precision_score(
            curr_label_precs, curr_pred_precs, average="micro", zero_division=0
        )
        accuracy_posts[i, :] = np.mean(curr_label_precs == curr_pred_precs)

    metrics = {
        f"{split}/state/f1_macro_precs": f1_macro_precs.mean().item(),
        f"{split}/state/f1_macro_posts": f1_macro_posts.mean().item(),
        f"{split}/state/f1_micro_precs": f1_micro_precs.mean().item(),
        f"{split}/state/f1_micro_posts": f1_micro_posts.mean().item(),
        f"{split}/state/recall_macro_precs": recall_macro_precs.mean().item(),
        f"{split}/state/recall_macro_posts": recall_macro_posts.mean().item(),
        f"{split}/state/recall_micro_precs": recall_micro_precs.mean().item(),
        f"{split}/state/recall_micro_posts": recall_micro_posts.mean().item(),
        f"{split}/state/precision_macro_precs": precision_macro_precs.mean().item(),
        f"{split}/state/precision_macro_posts": precision_macro_posts.mean().item(),
        f"{split}/state/precision_micro_precs": precision_micro_precs.mean().item(),
        f"{split}/state/precision_micro_posts": precision_micro_posts.mean().item(),
        f"{split}/state/accuracy_precs": accuracy_precs.mean().item(),
        f"{split}/state/accuracy_posts": accuracy_posts.mean().item(),
    }

    return metrics
