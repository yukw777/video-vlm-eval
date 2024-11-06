from transformers import AutoTokenizer, AutoConfig
from video_vlm_eval.model import TorchDType, Model
from video_vlm_eval.task import ZeroShotQA
from videollama2.mm_utils import (
    tokenizer_multimodal_token,
    KeywordsStoppingCriteria,
    process_video,
)
from videollama2.model.videollama2_qwen2 import Videollama2Qwen2ForCausalLM
from videollama2.model.videollama2_mistral import Videollama2MistralForCausalLM
from videollama2.model.videollama2_mixtral import Videollama2MixtralForCausalLM
from videollama2.constants import DEFAULT_VIDEO_TOKEN, NUM_FRAMES

from typing import Any
from functools import partial


class VideoLlama2Model(Model[dict[str, Any]]):
    def __init__(
        self,
        model_name_or_path: str,
        dtype: TorchDType | None = None,
        num_frames: int | None = None,
    ) -> None:
        super().__init__()
        self.dtype = dtype
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, use_fast=False
        )
        if self.tokenizer.pad_token is None and self.tokenizer.unk_token is not None:
            self.tokenizer.pad_token = self.tokenizer.unk_token
        config = AutoConfig.from_pretrained(model_name_or_path)
        model_type = config.model_type
        model_cls = Videollama2MistralForCausalLM
        if model_type in {"videollama2", "videollama2_mistral"}:
            model_cls = Videollama2MistralForCausalLM
        elif model_type in {"videollama2_mixtral"}:
            model_cls = Videollama2MixtralForCausalLM
        elif model_type in ["videollama2_qwen2"]:
            model_cls = Videollama2Qwen2ForCausalLM
        self.model = model_cls.from_pretrained(
            model_name_or_path,
            low_cpu_mem_usage=True,
            config=config,
            torch_dtype=self.dtype.value if self.dtype is not None else None,
        )
        vision_tower = self.model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()

        if self.dtype is not None:
            self.to(dtype=self.dtype.value)

        if num_frames is None:
            num_frames = (
                self.model.config.num_frames
                if hasattr(self.model.config, "num_frames")
                else NUM_FRAMES
            )
        self.processor = partial(
            process_video,
            processor=vision_tower.image_processor,
            aspect_ratio=None,
            num_frames=num_frames,
        )

        self.eval()

    def _build_prompt(self, datapoint: dict[str, Any]) -> str:
        messages = []
        if self.model.config.model_type in {
            "videollama2",
            "videollama2_mistral",
            "videollama2_mixtral",
        }:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        """<<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature."""
                        """\n"""
                        """If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>"""
                    ),
                }
            )
        messages.append(
            {
                "role": "user",
                "content": DEFAULT_VIDEO_TOKEN + "\n" + datapoint["question"],
            }
        )
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def preprocess(self, datapoint: dict[str, Any]) -> dict[str, Any]:
        datapoint["pixel_values"] = self.processor(str(datapoint.pop("video_path")))
        if self.dtype is not None:
            datapoint["pixel_values"] = datapoint["pixel_values"].to(self.dtype.value)
        datapoint["input_ids"] = tokenizer_multimodal_token(
            self._build_prompt(datapoint),
            self.tokenizer,
            DEFAULT_VIDEO_TOKEN,
            return_tensors="pt",
        ).unsqueeze(0)
        datapoint["attention_mask"] = (
            datapoint["input_ids"].ne(self.tokenizer.pad_token_id).long()
        )
        return datapoint

    def perform(self, batch: dict[str, Any], **gen_config) -> list[dict]:
        stopping_criteria = KeywordsStoppingCriteria(
            [self.tokenizer.eos_token], self.tokenizer, batch["input_ids"]
        )

        # NOTE: we do not support batch inference
        output_ids = self.model.generate(
            batch["input_ids"].squeeze(0),
            attention_mask=batch["attention_mask"].squeeze(0),
            images=[(batch["pixel_values"].squeeze(0), "video")],
            use_cache=True,
            stopping_criteria=[stopping_criteria],
            pad_token_id=self.tokenizer.eos_token_id,
            **gen_config,
        )

        return [
            {"answer": decoded.strip()}
            for decoded in self.tokenizer.batch_decode(
                output_ids, skip_special_tokens=True
            )
        ]


class VideoLlama2ZeroShotQAModel(VideoLlama2Model):
    def perform(self, batch: dict[str, Any], **gen_config) -> list[dict[str, str]]:
        return [
            {ZeroShotQA.pred_key: decoded["answer"]}
            for decoded in super().perform(batch, **gen_config)
        ]

    @property
    def result_keys(self) -> list[str]:
        return [ZeroShotQA.pred_key]
