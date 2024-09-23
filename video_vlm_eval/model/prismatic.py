import torch
from prismatic import load
from decord import VideoReader

from typing import Any
from video_vlm_eval.model import TorchDType, Model
from video_vlm_eval.model.utils import ORDINALS, EGOSCHEMA_OPTION_MAP
from video_vlm_eval.task import ZeroShotQA, MultipleChoice
from video_vlm_eval.task.video_chatgpt import VideoChatGPTConsistencyTask


class PrismaticModel(Model[dict[str, Any]]):
    def __init__(
        self,
        model_name_or_path: str,
        dtype: TorchDType | None,
        num_frame_samples: int | None = None,
        rope_scaling_type: str | None = None,
        rope_scaling_factor: float | None = None,
        llm_backbone_ckpt_path: str | None = None,
    ) -> None:
        super().__init__()
        self.model_name_or_path = model_name_or_path
        vision_backbone_kwargs = {}
        if num_frame_samples is not None:
            vision_backbone_kwargs["num_frame_samples"] = num_frame_samples
        llm_backbone_kwargs: dict[str, Any] = {}
        if rope_scaling_type is not None:
            llm_backbone_kwargs["rope_scaling_type"] = rope_scaling_type
        if rope_scaling_factor is not None:
            llm_backbone_kwargs["rope_scaling_factor"] = rope_scaling_factor
        self.model = load(
            model_name_or_path,
            vision_backbone_kwargs=vision_backbone_kwargs,
            llm_backbone_kwargs=llm_backbone_kwargs,
        )
        if llm_backbone_ckpt_path is not None:
            llm_backbone_state_dict = torch.load(llm_backbone_ckpt_path)["model"][
                "llm_backbone"
            ]
            self.model.llm_backbone.load_state_dict(llm_backbone_state_dict)
        self.to(dtype.value if dtype is not None else None)


class PrismaticZeroShotQAModel(PrismaticModel):
    def preprocess(self, datapoint: dict[str, Any]) -> dict[str, Any]:
        prompt_builder = self.model.get_prompt_builder()
        prompt_builder.add_turn("human", datapoint[ZeroShotQA.question_key])
        return {
            "pixel_values": self.model.vision_backbone.vision_transform(
                VideoReader(str(datapoint.pop("video_path")))
            ),
            "texts": prompt_builder.get_prompt(),
            **datapoint,
        }

    def perform(self, batch: dict[str, Any], **gen_config) -> list[dict[str, str]]:
        gen_texts = self.model.generate_batch(
            batch["pixel_values"], batch["texts"], **gen_config
        )
        return [{ZeroShotQA.pred_key: text} for text in gen_texts]  # type: ignore

    @property
    def result_keys(self) -> list[str]:
        return [ZeroShotQA.pred_key]


class PrismaticVideoChatGPTConsistencyModel(PrismaticModel):
    def preprocess(self, datapoint: dict[str, Any]) -> dict[str, Any]:
        prompt_builder_1 = self.model.get_prompt_builder()
        prompt_builder_1.add_turn(
            "human", datapoint[VideoChatGPTConsistencyTask.question_keys[0]]
        )
        prompt_builder_2 = self.model.get_prompt_builder()
        prompt_builder_2.add_turn(
            "human", datapoint[VideoChatGPTConsistencyTask.question_keys[1]]
        )
        return {
            "pixel_values": self.model.vision_backbone.vision_transform(
                VideoReader(str(datapoint.pop("video_path")))
            ),
            "texts_1": prompt_builder_1.get_prompt(),
            "texts_2": prompt_builder_2.get_prompt(),
            **datapoint,
        }

    def perform(self, batch: dict[str, Any], **gen_config) -> list[dict[str, str]]:
        gen_texts_1 = self.model.generate_batch(
            batch["pixel_values"], batch["texts_1"], **gen_config
        )
        gen_texts_2 = self.model.generate_batch(
            batch["pixel_values"], batch["texts_2"], **gen_config
        )
        return [
            {
                VideoChatGPTConsistencyTask.pred_keys[0]: text_1,  # type: ignore
                VideoChatGPTConsistencyTask.pred_keys[1]: text_2,  # type: ignore
            }
            for text_1, text_2 in zip(gen_texts_1, gen_texts_2, strict=True)
        ]

    @property
    def result_keys(self) -> list[str]:
        return VideoChatGPTConsistencyTask.pred_keys


class PrismaticEgoSchemaModel(PrismaticModel):
    """This model closely follows the official implementation of the mPLUG-Owl
    evaluation script for EgoSchema.

    More details can be found here:
    https://github.com/egoschema/EgoSchema/blob/fd7b3572f20e3297c28243bae940d00a092642ae/benchmarking/mPLUG-Owl/run_mplug.py
    """

    def preprocess(self, datapoint: dict[str, Any]) -> dict[str, Any]:
        return {
            "pixel_values": self.model.vision_backbone.vision_transform(
                VideoReader(str(datapoint.pop("video_path")))
            ),
            "texts": self._build_prompt(datapoint),
            **datapoint,
        }

    def _build_prompt(self, datapoint: dict[str, Any]) -> str:
        prompt_builder = self.model.get_prompt_builder()
        prompt_builder.add_turn(
            "human",
            'The video is shot from a first-person perspective and the "c" refers to camera wearer.\n'
            f"Question: {datapoint['question']}\n"
            "Options:\n"
            f"(A) {datapoint['option 0']}\n"
            f"(B) {datapoint['option 1']}\n"
            f"(C) {datapoint['option 2']}\n"
            f"(D) {datapoint['option 3']}\n"
            f"(E) {datapoint['option 4']}\n",
        )
        return prompt_builder.get_prompt() + "ASSISTANT: Answer:("

    def perform(self, batch: dict[str, Any], **gen_config) -> list[dict]:
        preds = self.model.generate_batch(
            batch["pixel_values"], batch["texts"], **gen_config
        )
        return [{MultipleChoice.pred_key: EGOSCHEMA_OPTION_MAP[pred]} for pred in preds]  # type: ignore

    @property
    def result_keys(self) -> list[str]:
        return [MultipleChoice.pred_key]


class PrismaticEgoSchemaNeedleHaystackModel(PrismaticEgoSchemaModel):
    def _build_prompt(self, datapoint: dict[str, Any]) -> str:
        if "scene_id" not in datapoint:
            return super()._build_prompt(datapoint)
        prompt_builder = self.model.get_prompt_builder()
        prompt_builder.add_turn(
            "human",
            'The video is shot from a first-person perspective and the "c" refers to camera wearer.\n'
            f'Given the question about the {ORDINALS[datapoint["scene_id"]]} scene.\n'
            f"Question: {datapoint['question']}\n"
            "Options:\n"
            f"(A) {datapoint['option 0']}\n"
            f"(B) {datapoint['option 1']}\n"
            f"(C) {datapoint['option 2']}\n"
            f"(D) {datapoint['option 3']}\n"
            f"(E) {datapoint['option 4']}\n",
        )
        return prompt_builder.get_prompt() + "ASSISTANT: Answer:("

    def preprocess(self, datapoint: dict[str, Any]) -> dict[str, Any]:
        extract_frames = self.model.vision_backbone.vision_transform.transforms[0]  # type: ignore
        original_num_frame_samples = extract_frames.num_frame_samples
        video_paths = datapoint.pop("video_paths")
        extract_frames.num_frame_samples = original_num_frame_samples // len(
            video_paths
        )
        # (C, T, H, W)
        pixel_values = torch.cat(
            [
                self.model.vision_backbone.vision_transform(
                    VideoReader(str(video_path))
                )  # type: ignore
                for video_path in video_paths
            ],
            dim=1,
        )
        extract_frames.num_frame_samples = original_num_frame_samples
        return {
            "pixel_values": pixel_values,
            "texts": self._build_prompt(datapoint),
            **datapoint,
        }
