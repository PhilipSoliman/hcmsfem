from numpy import ndarray
from torch import Tensor


class Operator(object):
    def apply(self, x: ndarray) -> ndarray:
        """        
        Apply the operator to a vector x.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def apply_gpu(self, x: Tensor) -> Tensor:
        """
        Apply the operator to a vector x on the GPU.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
