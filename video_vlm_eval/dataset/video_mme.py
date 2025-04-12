from datasets import load_dataset
from video_vlm_eval.dataset import Dataset
from typing import Any, Callable
from pathlib import Path


class VideoMMEDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        annotation_file: str,
        video_dir: str | None = None,
        preprocessor: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self.examples = load_dataset("parquet", data_files={"test": annotation_file})[
            "test"
        ]
        if video_dir is not None:
            video_dir_path = Path(video_dir)
            self.examples = self.examples.map(
                lambda ex: {
                    "video_path": str(video_dir_path / f"{ex['videoID']}.mp4"),
                    **ex,
                }
            )
        self._examples_by_id = {e[self.id_key]: e for e in self.examples}
        self.preprocessor = preprocessor

    def get_by_id(self, id: str) -> dict[str, Any]:
        return self._examples_by_id[id]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        datapoint = self.examples[idx]
        if self.preprocessor is not None:
            return self.preprocessor(datapoint)
        return datapoint

    def __len__(self) -> int:
        return len(self.examples)

    @property
    def columns(self) -> list[str]:
        return [
            "question_id",
            "duration",
            "domain",
            "sub_category",
            "task_type",
            "question",
            "options",
            "answer",
        ]

    @property
    def id_key(self) -> str:
        return "question_id"
