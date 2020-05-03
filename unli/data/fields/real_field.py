from typing import *
import logging
import torch
import math

from allennlp.data.fields import Field

logger = logging.getLogger(__name__)


class RealField(Field[torch.Tensor]):
    """
    A ``RealField`` contains a real-valued number.
    This field will be converted into a float tensor.
    """

    def __init__(self, value: float):
        self.value = value

    def get_padding_lengths(self) -> Dict[str, int]:
        return {}

    def as_tensor(self, padding_lengths: Dict[str, int]):
        tensor = torch.tensor(self.value, dtype=torch.float32)
        return tensor

    def empty_field(self) -> 'Field':
        return RealField(math.nan)

    def __str__(self) -> str:
        return f"RealField with value: {self.value}"


class IntField(Field[torch.Tensor]):
    """
    A ``IntField`` contains a integer-valued number.
    This field will be converted into a long tensor.
    """

    def __init__(self, value: int):
        self.value = value

    def get_padding_lengths(self) -> Dict[str, int]:
        return {}

    def as_tensor(self, padding_lengths: Dict[str, int]):
        tensor = torch.tensor(self.value, dtype=torch.int64)
        return tensor

    def empty_field(self) -> 'Field':
        return IntField(0)

    def __str__(self) -> str:
        return f"IntField with value: {self.value}"


