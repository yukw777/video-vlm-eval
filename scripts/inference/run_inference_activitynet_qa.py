import copy
import enum
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import wandb
from accelerate import Accelerator
from accelerate.utils import gather_object
from decord import VideoReader
from jsonargparse import CLI
from prismatic import load
from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import VideoTransform
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Disable Tokenizers Parallelism to Play Nice w/ PyTorch Multiprocessing DataLoaders
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class Preprocessor:
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


class ActivityNetQADataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        video_dirs: list[str],
        gt_file_question: str,
        preprocessor: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        """
        :param video_dirs: dirs that contain ActivityNet videos. this is a list to support different versions,
            e.g., ["v1-2/test", "v1-3/test"]
        :param gt_file_question: ActivityNet-QA question file, e.g., test_q.json
        :param tokenizer: pretrained tokenizer
        """
        with open(gt_file_question) as f:
            gt_questions = json.load(f)

        # figure out the video paths
        video_dir_paths = [Path(video_dir) for video_dir in video_dirs]
        self.examples: list[dict[str, Any]] = []
        for q in gt_questions:
            for p in video_dir_paths:
                vid_name = q["video_name"]
                vids = list(p.glob(f"*{vid_name}*"))
                if len(vids) == 0:
                    continue
                elif len(vids) > 1:
                    raise ValueError(
                        f"Multiple videos found for video {vid_name} in {p}"
                    )
                self.examples.append({"video_path": vids[0], **q})
                break
            else:
                raise ValueError(f"Couldn't find video {q['video_name']}")
        self.preprocessor = preprocessor

    def __getitem__(self, idx: int) -> dict[str, Any]:
        datapoint = self.examples[idx]
        if self.preprocessor is not None:
            return self.preprocessor(datapoint)
        return datapoint

    def __len__(self) -> int:
        return len(self.examples)


class TorchDType(enum.Enum):
    float16 = torch.float16
    bfloat16 = torch.bfloat16


def run(
    model_name_or_path: str,
    video_dirs: list[str],
    gt_file_question: str,
    dtype: TorchDType | None = None,
    num_dataloader_workers: int = 4,
    num_eval_steps: int | None = None,
    generation_config: str = "{}",
    wandb_project: str | None = None,
    random_seed: int = 42,
    print_gen_texts: bool = False,
) -> None:
    torch.manual_seed(random_seed)

    gen_config = json.loads(generation_config)
    if wandb_project is not None:
        accelerator = Accelerator(log_with="wandb")
        accelerator.init_trackers(
            wandb_project,
            config={
                "dtype": dtype.name if dtype is not None else None,
                "random_seed": random_seed,
                **gen_config,
            },
        )
        table = wandb.Table(
            columns=["video_name", "question_id", "question", "generated"]
        )
    else:
        accelerator = Accelerator()
        table = None

    model = load(model_name_or_path)
    model.to(dtype.value if dtype is not None else None)
    dataset = ActivityNetQADataset(
        video_dirs,
        gt_file_question,
        preprocessor=Preprocessor(
            model.vision_backbone.vision_transform,  # type: ignore
            model.get_prompt_builder(),
        ),
    )

    model, dataloader = accelerator.prepare(
        model,
        DataLoader(
            dataset,
            batch_size=1,
            num_workers=num_dataloader_workers,
            pin_memory=True,
        ),
    )

    module = model.module if isinstance(model, DistributedDataParallel) else model
    for i, batch in enumerate(
        tqdm(
            dataloader, desc="Generating", disable=not accelerator.is_local_main_process
        )
    ):
        if num_eval_steps is not None and i == num_eval_steps:
            break
        gathered_video_names = gather_object(batch["video_name"])
        gathered_question_ids = gather_object(batch["question_id"])
        gathered_questions = gather_object(batch["question"])
        gen_texts = module.generate_batch(
            batch["pixel_values"],
            batch["texts"],
            autocast_dtype=dtype.value if dtype is not None else None,
            **gen_config,
        )
        gathered_gen_texts = gather_object(gen_texts)
        if (
            accelerator.gradient_state.end_of_dataloader
            and accelerator.gradient_state.remainder > 0
        ):
            # we have some duplicates, so filter them out
            # this logic is from gather_for_metrics()
            gathered_video_names = gathered_video_names[
                : accelerator.gradient_state.remainder
            ]
            gathered_question_ids = gathered_question_ids[
                : accelerator.gradient_state.remainder
            ]
            gathered_questions = gathered_questions[
                : accelerator.gradient_state.remainder
            ]
            gathered_gen_texts = gathered_gen_texts[
                : accelerator.gradient_state.remainder
            ]
        if print_gen_texts:
            for gen_text in gathered_gen_texts:
                accelerator.print(f"Generated text: {gen_text}")
        if table is not None:
            for data in zip(
                gathered_video_names,
                gathered_question_ids,
                gathered_questions,
                gathered_gen_texts,
            ):
                table.add_data(*data)

    if table is not None:
        accelerator.log({"generated": table})

    accelerator.end_training()


if __name__ == "__main__":
    CLI(run, as_positional=False)
