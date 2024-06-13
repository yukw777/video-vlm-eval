import copy
from dataclasses import dataclass
from typing import Any

from decord import VideoReader
from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import VideoTransform


@dataclass
class PrismaticPreprocessor:
    video_transform: VideoTransform
    prompt_builder: PromptBuilder

    def __call__(self, datapoint: dict[str, Any]) -> dict[str, Any]:
        # we need to deep copy prompt_builder like this in order to avoid
        # pickling the whole VLM when using multiple dataloader workers.
        prompt_builder = copy.deepcopy(self.prompt_builder)
        prompt_builder.add_turn("human", datapoint["question"])
        return {
            "pixel_values": self.video_transform(
                VideoReader(str(datapoint.pop("video_path")))
            ),
            "texts": prompt_builder.get_prompt(),
            **datapoint,
        }
