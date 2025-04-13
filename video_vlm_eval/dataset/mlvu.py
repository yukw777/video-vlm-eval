import json
from video_vlm_eval.dataset import Dataset
from datasets import load_dataset
from typing import Any, Callable
from pathlib import Path


class MLVUMultipleChoiceDataset(Dataset):
    def __init__(
        self,
        annotation_file: str,
        video_path_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        preprocessor: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self.examples = load_dataset("parquet", data_files={"test": annotation_file})[
            "test"
        ]

        if video_path_fn is not None:
            self.examples = self.examples.map(video_path_fn)
        self.examples = self.examples.map(
            lambda ex: {"duration": str(ex.pop("duration")), **ex}
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
    def id_key(self) -> str:
        return "question_id"

    @property
    def columns(self) -> list[str]:
        return [
            "question_id",
            "video_name",
            "duration",
            "question",
            "answer",
            "candidates",
            "task_type",
        ]


class MLVUMultipleChoiceDevDataset(MLVUMultipleChoiceDataset):
    def __init__(
        self,
        annotation_file: str,
        video_dir: str | None = None,
        preprocessor: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        """
        :param video_dir: dir that contains videos, e.g., video/1_plotQA
        :param annotation_file: parquet file that contains all the multiple choice questions
        """
        if video_dir is None:
            super().__init__(annotation_file, preprocessor=preprocessor)
        else:
            task_video_dir_map = {
                video_dir.name.split("_", 1)[1]: video_dir
                for video_dir in Path(video_dir).iterdir()
            }

            def video_path_fn(ex: dict[str, Any]) -> dict[str, Any]:
                return {
                    "video_path": str(
                        task_video_dir_map[ex["task_type"]] / ex["video_name"]
                    ),
                    **ex,
                }

            super().__init__(
                annotation_file, video_path_fn=video_path_fn, preprocessor=preprocessor
            )


class MLVUMultipleChoiceTestDataset(MLVUMultipleChoiceDataset):
    def __init__(
        self,
        annotation_file: str,
        video_dir: str | None = None,
        preprocessor: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        """
        :param video_dir: dir that contains videos
        :param annotation_file: parquet file that contains all the multiple choice questions
        """
        if video_dir is None:
            super().__init__(annotation_file, preprocessor=preprocessor)
        else:
            video_dir_path = Path(video_dir)

            def video_path_fn(ex: dict[str, Any]) -> dict[str, Any]:
                return {
                    "video_path": str(video_dir_path / ex["video_name"]),
                    **ex,
                }

            super().__init__(
                annotation_file, video_path_fn=video_path_fn, preprocessor=preprocessor
            )


class MLVUGenerationDataset(Dataset[dict[str, Any]]):
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


class MLVUSSCDataset(MLVUGenerationDataset):
    @property
    def columns(self) -> list[str]:
        return ["question_id", "duration", "question", "answer", "scoring_points"]


class MLVUSummaryDataset(MLVUGenerationDataset):
    @property
    def columns(self) -> list[str]:
        return ["question_id", "duration", "question", "answer"]


class MLVUTestGenerationDataset(MLVUGenerationDataset):
    @property
    def columns(self) -> list[str]:
        return ["question_id", "video", "duration", "question", "question_type"]
