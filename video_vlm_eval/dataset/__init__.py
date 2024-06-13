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

    @abc.abstractmethod
    def get_by_id(self, id: str) -> T: ...

    @property
    @abc.abstractmethod
    def columns(self) -> tuple[str, ...]: ...

    @property
    @abc.abstractmethod
    def id_key(self) -> str: ...

    @property
    @abc.abstractmethod
    def question_key(self) -> str: ...

    @property
    @abc.abstractmethod
    def answer_key(self) -> str: ...
