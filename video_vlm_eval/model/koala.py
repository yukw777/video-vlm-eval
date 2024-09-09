import koala
import torch
import numpy as np

from pathlib import Path
from argparse import Namespace
from koala.common.config import Config
from koala.common.registry import registry
from video_vlm_eval.model import TorchDType, Model
from video_vlm_eval.task import MultipleChoice
from typing import Any
from decord import VideoReader
from PIL import Image


class KoalaModel(Model[dict[str, Any]]):
    def __init__(
        self,
        vicuna_path: str,
        minigpt_path: str,
        pretrained_weight_path: str,
        dtype: TorchDType | None,
        num_frames_per_clip: int = 16,
        num_segments: int = 4,
        hierarchical_agg_function: str = "without-top-final-global-prompts-region-segment-full-dis-spatiotemporal-prompts-attn-early-attn-linear-learned",
        pos_extending_factor: int | None = None,
    ) -> None:
        super().__init__()
        koala_root = Path(koala.__file__).parent
        cfg = Config(
            Namespace(
                cfg_path=str(
                    koala_root / "train_configs/video_aggregation_finetune.yaml"
                ),
                options=[
                    f"model.llama_model={vicuna_path}",
                    f"model.prompt_path={koala_root/'prompts/alignment_image.txt'}",
                    # f"model.ckpt={minigpt_path}",
                ],
            )
        )
        model_config = cfg.model_cfg
        model_cls = registry.get_model_class(model_config.arch)
        self.model = model_cls.from_config(model_config)

        self.model.num_frames_per_clip = num_frames_per_clip
        self.model.num_segments = num_segments
        self.model.hierarchical_agg_function = hierarchical_agg_function
        self.model.global_region_embed_weight = 1e-3

        self.model.initialize_visual_agg_function()

        vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
        self.vis_processor = registry.get_processor_class(
            vis_processor_cfg.name
        ).from_config(vis_processor_cfg)
        self.model.pos_extending_factor = pos_extending_factor

        pretrained_weights = torch.load(pretrained_weight_path, map_location="cpu")[
            "model_state_dict"
        ]
        pretrained_dict = {}
        for k, v in pretrained_weights.items():
            pretrained_dict[k.replace("module.", "")] = v

        model_dict = self.model.state_dict()
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)

        self.to(dtype.value if dtype is not None else None)


class KoalaEgoSchemaModel(KoalaModel):
    """This model closely follows the official implementation of Koala's
    evaluation script for EgoSchema.

    More details can be found here:
    https://github.com/rxtan2/Koala-video-llm/blob/273546773bc67cccd8cc58dfc781453b651a11c9/eval_qa_egoschema.py
    """

    def preprocess(self, datapoint: dict[str, Any]) -> dict[str, Any]:
        return {
            **self._extract_frames(str(datapoint.pop("video_path"))),
            **datapoint,
            **self._build_prompt(datapoint),
        }

    def _extract_frames(self, video_path: str) -> dict[str, torch.Tensor]:
        vr = VideoReader(video_path)
        global_clip_indices = np.linspace(
            0, len(vr) - 1, num=min(self.model.num_frames_per_clip, len(vr)), dtype=int
        )
        short_window_indices = np.linspace(
            0,
            len(vr) - 1,
            num=min(self.model.num_frames_per_clip * self.model.num_segments, len(vr)),
            dtype=int,
        )
        frames = vr.get_batch(
            np.concatenate((global_clip_indices, short_window_indices))
        ).asnumpy()

        global_processed_frames: list[torch.Tensor] = []
        for i in range(len(global_clip_indices)):
            global_processed_frames.append(
                self.vis_processor(Image.fromarray(frames[i]))
            )

        short_window_processed_frames: list[torch.Tensor] = []
        for i in range(len(global_clip_indices), len(short_window_indices)):
            short_window_processed_frames.append(
                self.vis_processor(Image.fromarray(frames[i]))
            )
        return {
            "global_video": torch.stack(global_processed_frames),
            "global_frame_attn_mask": torch.ones(len(global_clip_indices)),
            "segments_video": torch.stack(short_window_processed_frames),
            "segments_frame_attn_mask": torch.ones(len(short_window_indices)),
        }

    def _build_prompt(self, datapoint: dict[str, Any]) -> dict[str, Any]:
        prompt_dict = {}
        question = datapoint[MultipleChoice.question_key]
        for k in ["option 0", "option 1", "option 2", "option 3", "option 4"]:
            option = datapoint[k]
            prompt_builder = self.model.get_prompt_builder()
            prompt_builder.add_turn(
                "human",
                f"Given question '{question}, is answer '{option}' correct? "
                "Do you think that the answer to the given question is correct. "
                "Please answer yes or no.",
            )
            prompt_dict[f"{k}_prompt"] = prompt_builder.get_prompt()
        return prompt_dict

    def perform(self, batch: dict[str, Any], **gen_config) -> list[dict]:
        merged_video_embeds, merged_video_embeds_mask = (
            self.model.compute_merged_video_embeds(batch)
        )
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
