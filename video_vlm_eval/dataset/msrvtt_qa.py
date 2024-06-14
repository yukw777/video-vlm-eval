import json
from pathlib import Path
from typing import Any, Callable

from video_vlm_eval.dataset import Dataset


class MSRVTTQADataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        video_dir: str,
        annotations_file: str,
        preprocessor: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        """
        :param video_dir: dir that contains MSRVTT videos
        :param annotation: MSRVTT-QA annotation file, e.g., test_qa.json
        """
        with open(annotations_file) as f:
            annotations = json.load(f)

        video_id_to_path: dict[str, Path] = {}
        for video_path in Path(video_dir).iterdir():
            # remove "video" prefix
            video_id_to_path[video_path.stem[5:]] = video_path
        self.examples: list[dict[str, Any]] = []
        for ann in annotations:
            # we cast category_id, video_id and id to str so it wouldn't get turned into a tensor by the default collator.
            ann["category_id"] = str(ann["category_id"])
            ann["video_id"] = str(ann["video_id"])
            ann["id"] = str(ann["id"])

            self.examples.append(
                {"video_path": video_id_to_path[ann["video_id"]], **ann}
            )

        self._columns = [k for k in self.examples[0].keys() if k != "video_path"]
        self._id_key = "id"
        self._question_key = "question"
        self._answer_key = "answer"
        self._examples_by_id = {e[self._id_key]: e for e in self.examples}

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
        return self._columns

    @property
    def id_key(self) -> str:
        return self._id_key

    @property
    def question_key(self) -> str:
        return self._question_key

    @property
    def answer_key(self) -> str:
        return self._answer_key
