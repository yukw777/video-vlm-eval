import torch
from video_vlm_eval.model import TorchDType, Model
from video_vlm_eval.task import MultipleChoice
from video_vlm_eval.model.utils import ORDINALS

from tarsier.models.modeling_tarsier import TarsierForConditionalGeneration
from tarsier.dataset.processor import Processor

from typing import Any
from collections.abc import Callable


class TarsierModel(Model[dict[str, Any]]):
    def __init__(
        self,
        model_name_or_path: str,
        dtype: TorchDType | None,
        max_n_frames: int = 16,
        attn_implementation: str | None = None,
    ) -> None:
        super().__init__()
        self.processor = Processor(model_name_or_path, max_n_frames=max_n_frames)
        self.processor.tokenizer.padding_side = "left"
        self.model = TarsierForConditionalGeneration.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype.value if dtype is not None else None,
            attn_implementation=attn_implementation,
        )
        self.model.eval()


class TarsierEgoSchemaModel(TarsierModel):
    OPTION_MAP = {"A": "0", "B": "1", "C": "2", "D": "3", "E": "4"}

    def _build_prompt(self, datapoint: dict[str, Any]) -> str:
        return (
            'USER: <video> The video is shot from a first-person perspective and the "c" refers to camera wearer.\n'
            f"Question: {datapoint['question']}\n"
            "Options:\n"
            f"(A) {datapoint['option 0']}\n"
            f"(B) {datapoint['option 1']}\n"
            f"(C) {datapoint['option 2']}\n"
            f"(D) {datapoint['option 3']}\n"
            f"(E) {datapoint['option 4']}\n"
            "ASSISTANT: Answer: ("
        )

    def preprocess(self, datapoint: dict[str, Any]) -> dict[str, Any]:
        preprocessed = self.processor(
            self._build_prompt(datapoint),
            images=self.processor.load_images(str(datapoint.pop("video_path"))),
            edit_prompt=True,
        )
        preprocessed["input_ids"].squeeze_(dim=0)
        preprocessed.update(datapoint)
        return preprocessed

    @property
    def collate_fn(self) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
        def collate(datapoints: list[dict[str, Any]]) -> dict[str, Any]:
            pixel_values = torch.cat([d.pop("pixel_values") for d in datapoints])
            padded = self.processor.tokenizer.pad(
                [{"input_ids": d.pop("input_ids")} for d in datapoints]
            )
            inputs: dict[str, Any] = {
                k: [d[k] for d in datapoints] for k in datapoints[0].keys()
            }
            inputs.update(padded)
            inputs["pixel_values"] = pixel_values
            return inputs

        return collate

    def perform(self, batch: dict[str, Any], **gen_config) -> list[dict]:
        outputs = self.model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"],
            **gen_config,
        )
        preds: list[dict] = []
        for decoded in self.processor.tokenizer.batch_decode(outputs[:, -1]):
            if decoded in self.OPTION_MAP:
                pred = self.OPTION_MAP[decoded]
            elif "0" <= decoded <= "4":
                # Tarsier may erroneously generate numbers.
                # The original code translates the numbers into letters.
                # https://github.com/bytedance/tarsier/blob/9ff5567a8882cbcc81060f392bead76afb16e19d/evaluation/metrics/evaluate_qa_mc.py#L45-L47
                pred = self.OPTION_MAP[chr(int(decoded) + ord("A"))]
            else:
                # Tarsier generated an invalid answer, so just use it.
                # This will be marked wrong during evaluation.
                pred = decoded
            preds.append({MultipleChoice.pred_key: pred})
        return preds

    @property
    def result_keys(self) -> list[str]:
        return [MultipleChoice.pred_key]


class TarsierEgoSchemaNeedleHaystackModel(TarsierEgoSchemaModel):
    def _build_prompt(self, datapoint: dict[str, Any]) -> str:
        if "scene_id" not in datapoint:
            return super()._build_prompt(datapoint)

        return (
            'USER: <video> The video is shot from a first-person perspective and the "c" refers to camera wearer.\n'
            f"The following question is about the {ORDINALS[datapoint['scene_id']]} scene.\n"
            f"Question: {datapoint['question']}\n"
            "Options:\n"
            f"(A) {datapoint['option 0']}\n"
            f"(B) {datapoint['option 1']}\n"
            f"(C) {datapoint['option 2']}\n"
            f"(D) {datapoint['option 3']}\n"
            f"(E) {datapoint['option 4']}\n"
            "ASSISTANT: Answer: ("
        )

    def preprocess(self, datapoint: dict[str, Any]) -> dict[str, Any]:
        video_paths = datapoint.pop("video_paths")
        frames_per_video = self.processor.max_n_frames // len(video_paths)
        images = []
        for video_path in video_paths:
            images.extend(
                self.processor.load_images(str(video_path), n_frames=frames_per_video)
            )
        preprocessed = self.processor(
            self._build_prompt(datapoint), images=images, edit_prompt=True
        )
        preprocessed["input_ids"].squeeze_(dim=0)
        preprocessed.update(datapoint)
        return preprocessed
