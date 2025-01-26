import json
from video_vlm_eval.dataset import Dataset
from typing import Any, Callable
from pathlib import Path


class MLVUDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        video_dir: str,
        annotation_file: str,
        preprocessor: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        """
        :param video_dir: dir that contains videos, e.g., video/1_plotQA
        :param annotation_file: json annotation file, e.g., 1_plotQA.json
        """
        with open(annotation_file) as f:
            self.examples = json.load(f)

        for i, ex in enumerate(self.examples):
            if "question_id" not in ex:
                ex["question_id"] = f"{ex['question_type']}_{i}"
            ex["video_path"] = Path(video_dir) / ex["video"]

            # convert duration to string since this is not an input to the model
            ex["duration"] = str(ex["duration"])

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
        return ["question_id", "duration", "question", "answer", "candidates"]

    @property
    def id_key(self) -> str:
        return "question_id"
