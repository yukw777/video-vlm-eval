from prismatic import load
from decord import VideoReader

from typing import Any
from video_vlm_eval.model import TorchDType, Model
from video_vlm_eval.task import ZeroShotQA
from video_vlm_eval.task.video_chatgpt import VideoChatGPTConsistencyTask


class PrismaticModel(Model[dict[str, Any]]):
    def __init__(self, model_name_or_path: str, dtype: TorchDType | None) -> None:
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.model = load(model_name_or_path)
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
            for text_1, text_2 in zip(gen_texts_1, gen_texts_2)
        ]

    @property
    def result_keys(self) -> list[str]:
        return VideoChatGPTConsistencyTask.pred_keys
