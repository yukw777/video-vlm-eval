import json
from pathlib import Path
from typing import Any, Callable

from video_vlm_eval.dataset import Dataset


class ActivityNetQADataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        gt_file_question: str,
        gt_file_answer: str,
        video_dirs: list[str] | None = None,
        preprocessor: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        """
        :param video_dirs: dirs that contain ActivityNet videos. this is a list to support different versions,
            e.g., ["v1-2/test", "v1-3/test"]
        :param gt_file_question: ActivityNet-QA question file, e.g., test_q.json
        :param preprocessor: preprocessor
        """
        with open(gt_file_question) as f:
            gt_questions = json.load(f)
        with open(gt_file_answer) as f:
            gt_answers = json.load(f)

        # figure out the video paths
        video_name_to_path: dict[str, Path] = {}
        if video_dirs is not None:
            for video_dir in [Path(video_dir) for video_dir in video_dirs]:
                for video_path in video_dir.iterdir():
                    # video_name is the stem of the video path without the "v_" prefix
                    video_name_to_path[video_path.stem[2:]] = video_path
        self.examples: list[dict[str, Any]] = []
        for q, a in zip(gt_questions, gt_answers, strict=True):
            # we cast type to str so it wouldn't get turned into a tensor by the default collator.
            a["type"] = str(a["type"])
            if video_dirs is not None:
                self.examples.append(
                    {"video_path": video_name_to_path[q["video_name"]], **q, **a}
                )
            else:
                self.examples.append({**q, **a})

        self._columns = [k for k in self.examples[0].keys() if k != "video_path"]
        self._id_key = "question_id"
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
