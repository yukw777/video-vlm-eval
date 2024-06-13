import abc
from typing import Any, Callable, Generic, TypeVar

from torch.utils.data import Dataset as TorchDataset

T = TypeVar("T", covariant=True)


class Dataset(TorchDataset, Generic[T], abc.ABC):
    def __init__(
        self, preprocessor: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    ) -> None:
        self.preprocessor: Callable[[dict[str, Any]], dict[str, Any]] | None = (
            preprocessor
        )

    def set_preprocessor(
        self, preprocessor: Callable[[dict[str, Any]], dict[str, Any]]
    ) -> None:
        self.preprocessor = preprocessor

    @property
    @abc.abstractmethod
    def columns(self) -> tuple[str, ...]: ...
