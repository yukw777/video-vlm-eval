import json
from video_vlm_eval.dataset import Dataset
from typing import Any, Callable
from pathlib import Path


class MLVUDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        annotation_file: str,
        video_dir: str | None = None,
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
            if video_dir is not None:
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
    def id_key(self) -> str:
        return "question_id"


class MLVUMultipleChoiceDataset(MLVUDataset):
    @property
    def columns(self) -> list[str]:
        return ["question_id", "duration", "question", "answer", "candidates"]


class MLVUMultipleChoiceTestDataset(MLVUDataset):
    def __init__(
        self,
        annotation_file: str,
        video_dir: str | None = None,
        preprocessor: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(
            annotation_file, video_dir=video_dir, preprocessor=preprocessor
        )
        # candidates for count questions are int's for some reason,
        # and wandb sees it as a type mismatch as the candidates for other questions
        # are strings. So let's just cast them to strings.
        for e in self.examples:
            if not isinstance(e["candidates"][0], str):
                e["candidates"] = [str(cand) for cand in e["candidates"]]

    @property
    def columns(self) -> list[str]:
        return ["question_id", "question_type", "duration", "question", "candidates"]


class MLVUSSCDataset(MLVUDataset):
    @property
    def columns(self) -> list[str]:
        return ["question_id", "duration", "question", "answer", "scoring_points"]


class MLVUSummaryDataset(MLVUDataset):
    @property
    def columns(self) -> list[str]:
        return ["question_id", "duration", "question", "answer"]


class MLVUTestGenerationDataset(MLVUDataset):
    @property
    def columns(self) -> list[str]:
        return ["duration", "question", "question_type"]
