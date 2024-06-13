import json
from pathlib import Path
from typing import Any, Callable

from video_vlm_eval.dataset import Dataset


class ActivityNetQADataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        video_dirs: list[str],
        gt_file_question: str,
        gt_file_answer: str,
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
        video_dir_paths = [Path(video_dir) for video_dir in video_dirs]
        self.examples: list[dict[str, Any]] = []
        for q, a in zip(gt_questions, gt_answers):
            for p in video_dir_paths:
                vid_name = q["video_name"]
                vids = list(p.glob(f"*{vid_name}*"))
                if len(vids) == 0:
                    continue
                elif len(vids) > 1:
                    raise ValueError(
                        f"Multiple videos found for video {vid_name} in {p}"
                    )
                # we cast type to str so it wouldn't get turned into a tensor by the default collator.
                a["type"] = str(a["type"])
                self.examples.append({"video_path": vids[0], **q, **a})
                break
            else:
                raise ValueError(f"Couldn't find video {q['video_name']}")

        self._columns = [k for k in self.examples[0].keys() if k != "video_path"]
        self._id_key = "question_id"
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
