import json
from pathlib import Path
from typing import Any, Callable

from video_vlm_eval.dataset import Dataset


class ActivityNetQADataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        video_dirs: list[str],
        gt_file_question: str,
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

        # figure out the video paths
        video_dir_paths = [Path(video_dir) for video_dir in video_dirs]
        self.examples: list[dict[str, Any]] = []
        for q in gt_questions:
            for p in video_dir_paths:
                vid_name = q["video_name"]
                vids = list(p.glob(f"*{vid_name}*"))
                if len(vids) == 0:
                    continue
                elif len(vids) > 1:
                    raise ValueError(
                        f"Multiple videos found for video {vid_name} in {p}"
                    )
                self.examples.append({"video_path": vids[0], **q})
                break
            else:
                raise ValueError(f"Couldn't find video {q['video_name']}")
        self.preprocessor = preprocessor

    def __getitem__(self, idx: int) -> dict[str, Any]:
        datapoint = self.examples[idx]
        if self.preprocessor is not None:
            return self.preprocessor(datapoint)
        return datapoint

    def __len__(self) -> int:
        return len(self.examples)

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(k for k in self.examples[0].keys() if k not in {"video_path"})
