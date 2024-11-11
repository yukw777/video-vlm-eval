import json
from video_vlm_eval.dataset import Dataset
from typing import Any, Callable
from pathlib import Path


class VideoChatGPTGeneralDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        annotations_file: str,
        video_dir: str | None = None,
        preprocessor: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        """
        :param video_dir: dir that contains Video-ChatGPT test videos
        :param annotation: Video-ChatGPT general annotation file,
            e.g., generic_qa.json, temporal_qa.json
        """
        with open(annotations_file) as f:
            annotations = json.load(f)
        # add annotation id
        for i, ann in enumerate(annotations):
            ann["id"] = str(i)

        video_name_to_path: dict[str, Path] = {}
        if video_dir is not None:
            for video_path in Path(video_dir).iterdir():
                video_name_to_path[video_path.stem] = video_path
        self.examples: list[dict[str, Any]] = []
        for i, ann in enumerate(annotations):
            example = {
                "id": ann["id"],
                "question": ann["Q"],
                "answer": ann["A"],
                "video_name": ann["video_name"],
            }
            if video_dir is not None:
                example["video_path"] = video_name_to_path[ann["video_name"]]
            self.examples.append(example)

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
        return ["id", "video_name", "question", "answer"]

    @property
    def id_key(self) -> str:
        return "id"


class VideoChatGPTConsistencyDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        annotations_file: str,
        video_dir: str | None = None,
        preprocessor: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        """
        :param video_dir: dir that contains Video-ChatGPT test videos
        :param annotation: Video-ChatGPT consistency annotation file, e.g., consistency_qa.json
        """
        with open(annotations_file) as f:
            annotations = json.load(f)
        # add annotation id
        for i, ann in enumerate(annotations):
            ann["id"] = str(i)

        video_name_to_path: dict[str, Path] = {}
        if video_dir is not None:
            for video_path in Path(video_dir).iterdir():
                video_name_to_path[video_path.stem] = video_path
        self.examples: list[dict[str, Any]] = []
        for ann in annotations:
            if video_dir is not None:
                self.examples.append(
                    {"video_path": video_name_to_path[ann["video_name"]], **ann}
                )
            else:
                self.examples.append({**ann})

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
        return ["id", "video_name", "Q1", "Q2", "A"]

    @property
    def id_key(self) -> str:
        return "id"
