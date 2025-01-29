import json
from video_vlm_eval.dataset import Dataset
from typing import Any, Callable
from pathlib import Path


class MovieChat1KDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        gt_annotation_dir: str,
        video_dir: str | None = None,
        preprocessor: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        """
        :param video_dir: dir that contains videos
        :param gt_annotation_dir: dir that contains ground-truth annotation files
        """
        self.examples: list[dict[str, Any]] = []
        for ann_file in Path(gt_annotation_dir).iterdir():
            with open(ann_file) as f:
                anns = json.load(f)

            # we flatten out global and breakpoint questions
            # assign time as -1 for global questions
            video_name = Path(anns["info"]["video_path"]).stem
            questions = [
                {self.id_key: f"{video_name}_g_{i}", "time": -1, **g}
                for i, g in enumerate(anns["global"])
            ]
            questions += [
                {self.id_key: f"{video_name}_b_{i}", **b}
                for i, b in enumerate(anns["breakpoint"])
            ]
            if video_dir is not None:
                video_path = Path(video_dir) / anns["info"]["video_path"]
                for q in questions:
                    q["video_path"] = video_path
            self.examples.extend(questions)

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
        return ["question_id", "time", "question", "answer"]
