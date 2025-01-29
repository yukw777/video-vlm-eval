import numpy as np
import torch
from prismatic import load
from decord import VideoReader

from typing import Any, Callable

import torch.distributed
from torch.utils.data import default_collate
from video_vlm_eval.model import TorchDType, Model
from video_vlm_eval.model.utils import ORDINALS
from video_vlm_eval.task import ZeroShotQA, MultipleChoice
from video_vlm_eval.task.video_chatgpt import VideoChatGPTConsistencyTask


class PrismaticModel(Model[dict[str, Any]]):
    def __init__(
        self,
        model_name_or_path: str,
        dtype: TorchDType | None = None,
        num_frame_samples: int | None = None,
        rope_scaling_type: str | None = None,
        rope_scaling_factor: float | None = None,
        llm_backbone_ckpt_path: str | None = None,
        frames_per_seg: int | None = None,
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
        if torch.distributed.is_initialized():
            # timm has a race condition where if multiple processes try to
            # create the same model at the same time, it fails, so in the
            # torch distributed setting, let's have the rank 0 process
            # create the model first, and have others go after.
            rank = torch.distributed.get_rank()
            if rank == 0:
                self.model = load(
                    model_name_or_path,
                    vision_backbone_kwargs=vision_backbone_kwargs,
                    llm_backbone_kwargs=llm_backbone_kwargs,
                    frames_per_seg=frames_per_seg,
                )
            torch.distributed.barrier()
            if rank != 0:
                self.model = load(
                    model_name_or_path,
                    vision_backbone_kwargs=vision_backbone_kwargs,
                    llm_backbone_kwargs=llm_backbone_kwargs,
                    frames_per_seg=frames_per_seg,
                )
        else:
            self.model = load(
                model_name_or_path,
                vision_backbone_kwargs=vision_backbone_kwargs,
                llm_backbone_kwargs=llm_backbone_kwargs,
                frames_per_seg=frames_per_seg,
            )
        if llm_backbone_ckpt_path is not None:
            llm_backbone_state_dict = torch.load(llm_backbone_ckpt_path)["model"][
                "llm_backbone"
            ]
            self.model.llm_backbone.load_state_dict(llm_backbone_state_dict)
        self.to(dtype.value if dtype is not None else None)

    def preprocess(self, datapoint: dict[str, Any]) -> dict[str, Any]:
        prompt_builder = self.model.get_prompt_builder()
        prompt_builder.add_turn("human", datapoint["question"])
        return {
            "pixel_values": self.model.vision_backbone.vision_transform(
                VideoReader(str(datapoint.pop("video_path")))
            ),
            "texts": prompt_builder.get_prompt(),
        }

    def perform(self, batch: dict[str, Any], **gen_config) -> list[dict[str, str]]:
        gen_texts = self.model.generate_batch(
            batch["pixel_values"], batch["texts"], **gen_config
        )
        return [{"answer": text} for text in gen_texts]  # type: ignore


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
            **datapoint,
            **self._build_prompt(datapoint),
        }

    def _build_prompt(self, datapoint: dict[str, Any]) -> dict[str, Any]:
        prompt_dict = {}
        question = datapoint[MultipleChoice.question_key]
        for k in ["option 0", "option 1", "option 2", "option 3", "option 4"]:
            option = datapoint[k]
            prompt_builder = self.model.get_prompt_builder()
            prompt_builder.add_turn(
                "human",
                'The video is shot from a first-person perspective, and "c" refers to the camera wearer. '
                f'Given the question "{question}", is the answer "{option}" correct? '
                'Please answer only "yes" or "no".',
            )
            prompt_dict[f"{k}_prompt"] = prompt_builder.get_prompt()
        return prompt_dict

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


class PrismaticDirectAnswerEgoSchemaModel(PrismaticModel):
    OPTION_MAP = {"A": "0", "B": "1", "C": "2", "D": "3", "E": "4"}

    def preprocess(self, datapoint: dict[str, Any]) -> dict[str, Any]:
        return {
            "pixel_values": self.model.vision_backbone.vision_transform(
                VideoReader(str(datapoint.pop("video_path")))
            ),
            "texts": (
                'USER: The video is shot from a first-person perspective and the "c" refers to camera wearer.\n'
                f"Question: {datapoint['question']}\n"
                "Options:\n"
                f"(A) {datapoint['option 0']}\n"
                f"(B) {datapoint['option 1']}\n"
                f"(C) {datapoint['option 2']}\n"
                f"(D) {datapoint['option 3']}\n"
                f"(E) {datapoint['option 4']}\n"
                "ASSISTANT: Answer: ("
            ),
            **datapoint,
        }

    def perform(self, batch: dict[str, Any], **gen_config) -> list[dict[str, str]]:
        gen_texts = self.model.generate_batch(
            batch["pixel_values"], batch["texts"], **gen_config
        )
        preds: list[dict] = []
        for text in gen_texts:
            if text in self.OPTION_MAP:
                pred = self.OPTION_MAP[text]  # type: ignore
            elif "0" <= text <= "4":  # type: ignore
                # The model may erroneously generate numbers.
                # Tarsier's code translates the numbers into letters.
                # https://github.com/bytedance/tarsier/blob/9ff5567a8882cbcc81060f392bead76afb16e19d/evaluation/metrics/evaluate_qa_mc.py#L45-L47
                pred = self.OPTION_MAP[chr(int(text) + ord("A"))]  # type: ignore
            else:
                # The model generated an invalid answer, so just use it.
                # This will be marked wrong during evaluation.
                pred = text  # type: ignore
            preds.append({MultipleChoice.pred_key: pred})
        return preds

    @property
    def result_keys(self) -> list[str]:
        return [MultipleChoice.pred_key]


class PrismaticEgoSchemaNeedleHaystackModel(PrismaticEgoSchemaModel):
    def _build_prompt(self, datapoint: dict[str, Any]) -> dict[str, Any]:
        if "scene_id" not in datapoint:
            return super()._build_prompt(datapoint)
        prompt_dict = {}
        question = datapoint[MultipleChoice.question_key]
        for k in ["option 0", "option 1", "option 2", "option 3", "option 4"]:
            option = datapoint[k]
            prompt_builder = self.model.get_prompt_builder()
            prompt_builder.add_turn(
                "human",
                'The video is shot from a first-person perspective, and "c" refers to the camera wearer. '
                f'Given the question about the {ORDINALS[datapoint["scene_id"]]} scene, "{question}", '
                f'is the answer "{option}" correct? '
                'Please answer only "yes" or "no".',
            )
            prompt_dict[f"{k}_prompt"] = prompt_builder.get_prompt()
        return prompt_dict

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
            **datapoint,
            **self._build_prompt(datapoint),
        }


class PrismaticMLVUMultipleChoiceModel(PrismaticEgoSchemaModel):
    def _build_prompt(self, datapoint: dict[str, Any]) -> dict[str, Any]:
        prompt_dict = {}
        question = datapoint[MultipleChoice.question_key]
        for i, cand in enumerate(datapoint["candidates"]):
            prompt_builder = self.model.get_prompt_builder()
            prompt_builder.add_turn(
                "human",
                f'Given the question "{question}", is the answer "{cand}" correct? '
                'Please answer only "yes" or "no".',
            )
            prompt_dict[f"candidate_{i}_prompt"] = prompt_builder.get_prompt()
        return prompt_dict

    def perform(self, batch: dict[str, Any], **gen_config) -> list[dict]:
        # confidence level for each option (batch, num_cands)
        num_cands = len(batch["candidates"][0])
        confidence = np.zeros((batch["pixel_values"].size(0), num_cands))
        for i in range(num_cands):
            _, batch_gen_probs = self.model.generate_batch(
                batch["pixel_values"],
                batch[f"candidate_{i}_prompt"],
                return_string_probabilities=["Yes", "No"],
                **gen_config,
            )
            for j, gen_probs in enumerate(batch_gen_probs):
                confidence[j][i] = gen_probs[0]  # type: ignore

        batch_pred = confidence.argmax(axis=1)

        return [
            {MultipleChoice.pred_key: cands[pred]}
            for cands, pred in zip(
                batch["candidates"], batch_pred.tolist(), strict=True
            )
        ]

    @property
    def result_keys(self) -> list[str]:
        return [MultipleChoice.pred_key]

    @property
    def collate_fn(self) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
        def collate(datapoints: list[dict[str, Any]]) -> dict[str, Any]:
            # the default collator transposes lists of lists, so let's collate "candidates" manually
            batch_candidates = [datapoint.pop("candidates") for datapoint in datapoints]
            collated = default_collate(datapoints)
            collated["candidates"] = batch_candidates
            return collated

        return collate


class PrismaticMLVUGenerationModel(PrismaticZeroShotQAModel):
    @property
    def collate_fn(self) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
        def collate(datapoints: list[dict[str, Any]]) -> dict[str, Any]:
            # the lengths of scoring_points are variable, so let's collate "candidates" manually if they exist
            if "scoring_points" in datapoints[0]:
                batch_candidates = [
                    datapoint.pop("scoring_points") for datapoint in datapoints
                ]
                collated = default_collate(datapoints)
                collated["scoring_points"] = batch_candidates
                return collated
            return default_collate(datapoints)

        return collate


class PrismaticMovieChat1KModel(PrismaticZeroShotQAModel):
    def __init__(
        self,
        model_name_or_path: str,
        dtype: TorchDType | None = None,
        num_frame_samples: int | None = None,
        rope_scaling_type: str | None = None,
        rope_scaling_factor: float | None = None,
        llm_backbone_ckpt_path: str | None = None,
        frames_per_seg: int | None = None,
    ):
        super().__init__(
            model_name_or_path,
            dtype=dtype,
            num_frame_samples=num_frame_samples,
            rope_scaling_type=rope_scaling_type,
            rope_scaling_factor=rope_scaling_factor,
            llm_backbone_ckpt_path=llm_backbone_ckpt_path,
            frames_per_seg=frames_per_seg,
        )
        self.extract_frames = (
            self.model.vision_backbone.vision_transform.transforms.pop(0)  # type: ignore
        )

    def preprocess(self, datapoint: dict[str, Any]) -> dict[str, Any]:
        prompt_builder = self.model.get_prompt_builder()
        prompt_builder.add_turn("human", datapoint[ZeroShotQA.question_key])
        if datapoint["time"] == -1:
            # global question, so extract frames from the whole video
            pixel_values = self.model.vision_backbone.vision_transform(
                self.extract_frames(VideoReader(str(datapoint.pop("video_path"))))
            )
        else:
            # breakpoint question, so extract frames from the interval centered around "time"
            half = self.extract_frames.num_frame_samples // 2
            pixel_values = self.model.vision_backbone.vision_transform(
                self.extract_frames(
                    VideoReader(str(datapoint.pop("video_path"))),
                    time_interval=(datapoint["time"] - half, datapoint["time"] + half),
                )
            )
        return {
            "pixel_values": pixel_values,
            "texts": prompt_builder.get_prompt(),
            **datapoint,
        }
