import csv
from pathlib import Path
from typing import Any, Callable

from video_vlm_eval.dataset import Dataset


class TGIFQAFrameDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        video_dir: str,
        frame_annotation_file: str,
        preprocessor: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        """Note that this dataset is only for the "Frame" annotations from
        TGIF-QA, which is the annotations used by Video-ChatGPT and Video-
        LLaVA. See the links below for more information:

        - https://github.com/mbzuai-oryx/Video-ChatGPT/issues/65#issuecomment-1902476327
        - https://github.com/PKU-YuanGroup/Video-LLaVA/issues/37#issuecomment-1900067521

        :param video_dir: dir that contains converted TGIF videos
        :param frame_annotation_file: TGIF-QA Frame annotations file
        """
        gif_name_to_path: dict[str, Path] = {}
        for video_path in Path(video_dir).iterdir():
            gif_name_to_path[video_path.stem] = video_path

        self.examples: list[dict[str, Any]] = []
        with open(frame_annotation_file, newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for ann in reader:
                self.examples.append(
                    {"video_path": gif_name_to_path[ann["gif_name"]], **ann}
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
    def columns(self) -> list[str]:
        return ["gif_name", "question", "answer", "vid_id", "key"]

    @property
    def id_key(self) -> str:
        return "vid_id"
