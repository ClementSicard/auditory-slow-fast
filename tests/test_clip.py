import clip
import torch

from typing import Dict, Any
import argparse
from loguru import logger


def test() -> str:
    text = "a photo of a cat"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/32", device=device, download_root="/scratch/cs7561/clip")

    logger.info(f"Running for text '{text}'")

    tokenized_text = clip.tokenize([text]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(tokenized_text)

    logger.success(f"Embedding shape: {text_features.shape}")
    logger.debug(f"Embeddings: {text_features}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test CLIP")
    parser.add_argument("test", type=str)
    args = vars(parser.parse_args())

    test(text=args["test"])
