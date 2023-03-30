import os
import numpy as np
import librosa
import torch


import sys
sys.path.append(os.path.abspath(os.path.join(__file__, '../..')))
# pip install librosa simplejson psutil
import slowfast.utils.checkpoint as cu
from slowfast.models import build_model
# from slowfast.datasets.utils import pack_pathway_output
from slowfast.utils.parser import load_config, parse_args

def read_labels(fname):
    import csv
    with open(fname, newline='') as csvfile:
        return list(csv.DictReader(csvfile))

def main():
    args = parse_args()
    cfg = load_config(args)
    cfg.NUM_GPUS = int(bool(torch.cuda.is_available()))
    cfg.TEST.CHECKPOINT_FILE_PATH = 'checkpoints/SLOWFAST_EPIC.pyth'
    print(cfg.TEST.CHECKPOINT_FILE_PATH)
    model = build_model(cfg)
    cu.load_test_checkpoint(cfg, model)
    print(model)

    inputs = load_audio(cfg, args.path)
    inputs = [x[None] for x in inputs]
    desc(inputs)
    desc(*inputs)
    with torch.no_grad():
        preds = model(inputs)
    desc(*preds)
    # [1, 4365] => 4365/(96 + 1) == 45   - len(verb_classes) == 96
    # [1, 13500] => 13500/(299 + 1) == 45   - len(noun_classes) == 299

    verb_pred = preds[0].numpy().reshape((-1, 97))
    noun_pred = preds[1].numpy().reshape((-1, 300))
    print(verb_pred.shape, noun_pred.shape)

    verbs = read_labels('EPIC_100_verb_classes.csv')
    nouns = read_labels('EPIC_100_noun_classes.csv')

    import matplotlib.pyplot as plt
    plt.figure(figsize=(15, 10))
    plt.subplot(121)
    plt.imshow(verb_pred.T)
    N = verb_pred.shape[1]-1
    yticks = np.linspace(0, N, 40, dtype=int)
    plt.yticks(yticks, np.array([d['key'] for d in verbs])[yticks])
    plt.subplot(122)
    plt.imshow(noun_pred.T)
    N = noun_pred.shape[1]-1
    yticks = np.linspace(0, N, 40, dtype=int)
    plt.yticks(yticks, np.array([d['key'] for d in nouns])[yticks])
    plt.show()


def load_audio(cfg, path, eps=1e-6):
    window_size = cfg.AUDIO_DATA.WINDOW_LENGTH
    step_size = cfg.AUDIO_DATA.HOP_LENGTH
    nperseg = int(round(window_size * cfg.AUDIO_DATA.SAMPLING_RATE / 1e3))
    noverlap = int(round(step_size * cfg.AUDIO_DATA.SAMPLING_RATE / 1e3))
    
    y, sr = librosa.load(path, sr=cfg.AUDIO_DATA.SAMPLING_RATE, mono=False)

    spec = librosa.stft(
        y, n_fft=2048,
        window='hann',
        hop_length=noverlap,
        win_length=nperseg,
        pad_mode='constant')
    mel_basis = librosa.filters.mel(sr=sr, n_fft=2048, n_mels=128, htk=True, norm=None)
    spec = np.dot(mel_basis, np.abs(spec))
    spec = np.log(spec + eps)

    npad = max(0, cfg.AUDIO_DATA.NUM_FRAMES - spec.shape[0])
    spec = np.pad(spec, ((0, npad), (0, 0)), 'edge')
    spec = torch.tensor(spec)[None]
    spec = pack_pathway_output(cfg, spec)
    return spec

def desc(*xs):
    for x in xs:
        try:
            nonzerodim = all(x.shape)
            print(x.shape, x.dtype, x.min() if nonzerodim else None, x.max() if nonzerodim else None)
        except AttributeError:
            l = ''
            try:
                l = len(x)
            except Exception:
                pass
            print(type(x).__name__, l)


def parse_args():
    """
    Parse the following arguments for a default parser for PySlowFast users.
    Args:
        shard_id (int): shard id for the current machine. Starts from 0 to
            num_shards - 1. If single machine is used, then set shard id to 0.
        num_shards (int): number of shards using by the job.
        init_method (str): initialization method to launch the job with multiple
            devices. Options includes TCP or shared file-system for
            initialization. details can be find in
            https://pytorch.org/docs/stable/distributed.html#tcp-initialization
        cfg (str): path to the config file.
        opts (argument): provide addtional options from the command line, it
            overwrites the config loaded from file.
    """
    import argparse
    parser = argparse.ArgumentParser(description="Provide SlowFast audio training and testing pipeline.")
    parser.add_argument("path", help="The audio file")
    parser.add_argument(
        "--shard_id",
        help="The shard id of current node, Starts from 0 to num_shards - 1",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--num_shards",
        help="Number of shards using by the job",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--init_method",
        help="Initialization method, includes TCP or shared file-system",
        default="tcp://localhost:9999",
        type=str,
    )
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Path to the config file",
        default='configs/EPIC-KITCHENS/SLOWFAST_R50.yaml',
    )
    parser.add_argument(
        "opts",
        help="See slowfast/config/defaults.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    if len(sys.argv) == 1:
        parser.print_help()
    return parser.parse_args()



def pack_pathway_output(cfg, spectrogram):
    """
    Prepare output as a list of tensors. Each tensor corresponding to a
    unique pathway.
    Args:
        spectrogram (tensor): frames of spectrograms sampled from the complete spectrogram. The
            dimension is `channel` x `num frames` x `num frequencies`.
    Returns:
        spectrogram_list (list): list of tensors with the dimension of
            `channel` x `num frames` x `num frequencies`.
    """
    if cfg.MODEL.ARCH in cfg.MODEL.SINGLE_PATHWAY_ARCH:
        spectrogram_list = [spectrogram]
    elif cfg.MODEL.ARCH in cfg.MODEL.MULTI_PATHWAY_ARCH:
        fast_pathway = spectrogram
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            spectrogram, 1,
            torch.linspace(0, spectrogram.shape[1] - 1, spectrogram.shape[1] // cfg.SLOWFAST.ALPHA).long(),
        )
        spectrogram_list = [slow_pathway, fast_pathway]
    else:
        raise NotImplementedError(
            "Model arch {} is not in {}".format(
                cfg.MODEL.ARCH, cfg.MODEL.SINGLE_PATHWAY_ARCH + cfg.MODEL.MULTI_PATHWAY_ARCH))
    return spectrogram_list


if __name__ == '__main__':
    main()