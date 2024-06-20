import json
from video_vlm_eval.dataset import Dataset
from typing import Any, Callable
from pathlib import Path


class VideoChatGPTConsistencyDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        video_dir: str,
        annotations_file: str,
        preprocessor: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        """
        :param video_dir: dir that contains Video-ChatGPT test videos
        :param annotation: Video-ChatGPT consistency annotation file, e.g., consistency_qa.json
        """
        with open(annotations_file) as f:
            annotations = json.load(f)

        video_id_to_path: dict[str, Path] = {}
        for video_path in Path(video_dir).iterdir():
            video_id_to_path[video_path.stem] = video_path
        self.examples: list[dict[str, Any]] = []
        for ann in annotations:
            self.examples.append(
                {"video_path": video_id_to_path[ann["video_name"]], **ann}
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
        return ["Q1", "Q2", "A", "video_name"]

    @property
    def id_key(self) -> str:
        return "video_name"
