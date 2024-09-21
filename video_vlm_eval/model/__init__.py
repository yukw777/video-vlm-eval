import abc
from typing import TypeVar, Generic
from collections.abc import Callable
import enum
import torch
from torch import nn


class TorchDType(enum.Enum):
    float16 = torch.float16
    bfloat16 = torch.bfloat16


T = TypeVar("T")


class Model(nn.Module, Generic[T], abc.ABC):
    @abc.abstractmethod
    def perform(self, batch: T, **kwargs) -> list[dict]: ...

    @abc.abstractmethod
    def preprocess(self, datapoint) -> T: ...

    @property
    @abc.abstractmethod
    def result_keys(self) -> list[str]: ...

    @property
    def collate_fn(self) -> Callable[[list[T]], T] | None:
        return None
