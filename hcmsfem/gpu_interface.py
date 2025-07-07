from typing import Optional

import numpy as np
import scipy.sparse as sp
import torch as tr

from hcmsfem.logger import LOGGER


class GPUInterface:
    """
    Interface for sending matrices to the GPU.
    """

    DEVICE = "cuda" if tr.cuda.is_available() else "cpu"
    AVAILABLE = False
    AVAILABILTY_MSG = (
        "CUDA is not available. Please check if [link=https://developer.nvidia.com/cuda-downloads]CUDA[/link] is installed and configured correctly."
        "[link=https://pytorch.org/get-started/locally/]PyTorch CUDA installation guide[/link]"
    )

    def __init__(self):
        if not self.available():
            LOGGER.warning(self.AVAILABILTY_MSG)
        self.AVAILABLE = True

    def available(self) -> bool:
        """
        Check if CUDA is available.

        Returns:
            bool: True if CUDA is available, False otherwise.
        """
        if self.DEVICE == "cpu":
            return False
        return True

    def send_matrix(
        self, mat: sp.csc_matrix | sp.csr_matrix, dense: bool = False
    ) -> tr.Tensor:
        """
        Send a sparse matrix to the GPU as a sparse COO tensor or dense tensor.

        Args:
            mat (sp.csc_matrix | sp.csr_matrix): The sparse matrix to send.
            dense (bool): If True, convert the sparse matrix to a dense tensor.

        Returns:
            tr.Tensor: The matrix on the specified device.
        """
        if not dense:
            coo = mat.tocoo()
            return tr.sparse_coo_tensor(np.array([coo.row, coo.col]), coo.data, coo.shape, device=GPUInterface.DEVICE)  # type: ignore
        else:
            return tr.tensor(mat.toarray(), dtype=tr.float64, device=GPUInterface.DEVICE)  # type: ignore

    def send_array(
        self, arr: np.ndarray, dtype: Optional[tr.dtype | np.dtype] = np.float64
    ) -> tr.Tensor:
        """
        Send a NumPy array to the GPU as a tensor.

        Args:
            arr (np.ndarray): The array to send.
            dtype (Optional[tr.dtype]): The data type of the tensor (default is None).

        Returns:
            tr.Tensor: The array as a tensor on the specified device.
        """
        return tr.tensor(arr, dtype=dtype, device=self.DEVICE)

    @staticmethod
    def retrieve_array(tensor: tr.Tensor) -> np.ndarray:
        """
        Retrieve a tensor from the GPU and convert it to a NumPy array.

        Args:
            tensor (tr.Tensor): The tensor to retrieve.

        Returns:
            np.ndarray: The tensor as a NumPy array.
        """
        return tensor.cpu().numpy()


if __name__ == "__main__":
    gpu_interface = GPUInterface()
