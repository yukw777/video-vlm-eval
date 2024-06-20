import json
from pathlib import Path
from typing import Any, Callable

from video_vlm_eval.dataset import Dataset


class MSVDQADataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        video_dir: str,
        video_mapping_file: str,
        annotations_file: str,
        preprocessor: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        """
        :param video_dir: dir that contains MSVD videos
        :param video_mapping: file that specifies mapping b/w MSVD videos and MSVD-QA annotations
        :param annotation: MSVD-QA annotation file, e.g., test_qa.json
        """
        with open(video_mapping_file) as f:
            self.video_mapping = {}
            for line in f:
                msvd_vid, msvd_qa_vid = line.split()
                # remove "vid" from msvd_qa_vid
                self.video_mapping[msvd_qa_vid[3:]] = msvd_vid

        with open(annotations_file) as f:
            annotations = json.load(f)

        msvd_vid_to_path: dict[str, Path] = {}
        for video_path in Path(video_dir).iterdir():
            msvd_vid_to_path[video_path.stem] = video_path
        self.examples: list[dict[str, Any]] = []
        for ann in annotations:
            # we cast video_id and id to str so it wouldn't get turned into a tensor by the default collator.
            ann["video_id"] = str(ann["video_id"])
            ann["id"] = str(ann["id"])

            msvd_qa_vid = ann["video_id"]
            msvd_vid = self.video_mapping[msvd_qa_vid]
            self.examples.append({"video_path": msvd_vid_to_path[msvd_vid], **ann})

        self._columns = [k for k in self.examples[0].keys() if k != "video_path"]
        self._id_key = "id"
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
