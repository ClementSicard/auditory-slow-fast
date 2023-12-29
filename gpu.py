import argparse
import time

import torch
from loguru import logger


def gpu_stress_test(n: int):
    # Check if CUDA is available
    if torch.cuda.is_available():
        # Set the device to GPU
        device = torch.device("cuda")
        logger.info("Running GPU stress test on:", torch.cuda.get_device_name(device))

        # Create large random matrices
        A = torch.randn(n, n, device=device)
        B = torch.randn(n, n, device=device)

        # Perform matrix multiplication repeatedly
        while True:
            torch.matmul(A, B)

            # Optional: Sleep for a short duration
            time.sleep(0.1)  # Adjust the sleep time as needed

    else:
        logger.error("CUDA not available. Please run this on a machine with a CUDA-capable GPU.")
        exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", help="Size of the matrix to be multiplied", type=int, default=6200)
    args = parser.parse_args()

    gpu_stress_test(args.n)
