import json
from video_vlm_eval.dataset import Dataset
from typing import Any, Callable
from pathlib import Path


class EgoSchemaDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        video_dir: str,
        question_file: str,
        answer_file: str | None = None,
        preprocessor: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        """
        :param video_dir: dir that contains EgoSchema videos
        :param question_file: EgoSchema question file, e.g., questions.json
        :param answer_file: EgoSchema answer file, e.g., subset_answers.json
        """
        with open(question_file) as f:
            questions = json.load(f)
        answers = {}
        if answer_file is not None:
            with open(answer_file) as f:
                answers = json.load(f)

        video_name_to_path: dict[str, Path] = {}
        for video_path in Path(video_dir).iterdir():
            video_name_to_path[video_path.stem] = video_path

        self.examples: list[dict[str, Any]] = []
        for q in questions:
            self.examples.append(
                {
                    "video_path": video_name_to_path[q[self.id_key]],
                    "answer": str(answers.get(q["q_uid"], "")),
                    **q,
                }
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
        return [
            "q_uid",
            "google_drive_id",
            "question",
            "answer",
            "option 0",
            "option 1",
            "option 2",
            "option 3",
            "option 4",
        ]

    @property
    def id_key(self) -> str:
        return "q_uid"


class EgoSchemaNeedleHaystackDataset(EgoSchemaDataset):
    def __init__(
        self,
        video_dir: str,
        question_file: str,
        needle_haystack_mapping_file: str,
        answer_file: str | None = None,
        preprocessor: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        add_scene: bool = True,
    ) -> None:
        """
        :param video_dir: dir that contains EgoSchema videos
        :param question_file: EgoSchema question file, e.g., questions.json
        :param needle_haystack_mapping_file: Needle in a haystack mapping file for videos
        :param answer_file: EgoSchema answer file, e.g., subset_answers.json
        :param add_scene: whether to add the scene reference in the questions
        """
        with open(question_file) as f:
            questions = json.load(f)
        answers = {}
        if answer_file is not None:
            with open(answer_file) as f:
                answers = json.load(f)

        with open(needle_haystack_mapping_file) as f:
            needle_haystack_mapping = json.load(f)

        video_name_to_path: dict[str, Path] = {}
        for video_path in Path(video_dir).iterdir():
            video_name_to_path[video_path.stem] = video_path

        self.examples: list[dict[str, Any]] = []
        for q in questions:
            q_uid = q[self.id_key]
            if add_scene:
                scene_id = needle_haystack_mapping[q_uid].index(q_uid)
                q["scene_id"] = scene_id
            self.examples.append(
                {
                    "video_paths": [
                        video_name_to_path[vid_id]
                        for vid_id in needle_haystack_mapping[q_uid]
                    ],
                    "answer": str(answers.get(q_uid, "")),
                    **q,
                }
            )

        self._examples_by_id = {e[self.id_key]: e for e in self.examples}
        self.preprocessor = preprocessor
