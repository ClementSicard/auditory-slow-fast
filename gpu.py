import torch
import time
from loguru import logger


def gpu_stress_test():
    # Check if CUDA is available
    if torch.cuda.is_available():
        # Set the device to GPU
        device = torch.device("cuda")
        logger.info("Running GPU stress test on:", torch.cuda.get_device_name(device))

        # Create large random matrices
        matrix_size = 6200  # You can adjust this size
        A = torch.randn(matrix_size, matrix_size, device=device)
        B = torch.randn(matrix_size, matrix_size, device=device)

        # Perform matrix multiplication repeatedly
        while True:
            torch.matmul(A, B)

            # Optional: Sleep for a short duration
            time.sleep(0.1)  # Adjust the sleep time as needed

    else:
        logger.error("CUDA not available. Please run this on a machine with a CUDA-capable GPU.")
        exit(1)


if __name__ == "__main__":
    gpu_stress_test()
