import numpy as np
from prismatic import load
from decord import VideoReader

from typing import Any
from video_vlm_eval.model import TorchDType, Model
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
        question = datapoint[MultipleChoice.question_key]

        preprocessed = {
            "pixel_values": self.model.vision_backbone.vision_transform(
                VideoReader(str(datapoint.pop("video_path")))
            ),
            **datapoint,
        }
        for k in ["option 0", "option 1", "option 2", "option 3", "option 4"]:
            option = datapoint[k]
            prompt_builder = self.model.get_prompt_builder()
            prompt_builder.add_turn(
                "human",
                f"Given question '{question}, is answer '{option}' correct? "
                "Do you think that the answer to the given question is correct. "
                "Please answer yes or no.",
            )
            preprocessed[f"{k}_prompt"] = prompt_builder.get_prompt()
        return preprocessed

    def perform(self, batch: dict[str, Any], **gen_config) -> list[dict]:
        # confidence level for each option (batch, num_option)
        confidence = np.zeros((batch["pixel_values"].size(0), 5))
        for o, k in enumerate(
            ["option 0", "option 1", "option 2", "option 3", "option 4"]
        ):
            _, batch_gen_probs = self.model.generate_batch(
                batch["pixel_values"],
                batch[f"{k}_prompt"],
                return_string_probabilities=["Yes", "No"],
                **gen_config,
            )
            for i, gen_probs in enumerate(batch_gen_probs):
                confidence[i][o] = gen_probs[0]  # type: ignore

        batch_pred = confidence.argmax(axis=1)

        return [{MultipleChoice.pred_key: pred} for pred in batch_pred.tolist()]

    @property
    def result_keys(self) -> list[str]:
        return [MultipleChoice.pred_key]
