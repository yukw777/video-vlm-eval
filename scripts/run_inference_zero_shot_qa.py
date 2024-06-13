import csv
import enum
import os

import torch
from accelerate import Accelerator
from accelerate.utils import gather_object, set_seed, tqdm
from jsonargparse import CLI
from prismatic import load
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from video_vlm_eval import Dataset, PrismaticPreprocessor

# Disable Tokenizers Parallelism to Play Nice w/ PyTorch Multiprocessing DataLoaders
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class TorchDType(enum.Enum):
    float16 = torch.float16
    bfloat16 = torch.bfloat16


def run(
    model_name_or_path: str,
    dataset: Dataset,
    dtype: TorchDType | None = None,
    num_dataloader_workers: int = 4,
    num_eval_steps: int | None = None,
    gen_config: dict | None = None,
    wandb_project: str | None = None,
    random_seed: int = 42,
    out_file_name: str | None = None,
) -> None:
    set_seed(random_seed)

    if gen_config is None:
        gen_config = {}
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
    else:
        accelerator = Accelerator()

    model = load(model_name_or_path)
    model.to(dtype.value if dtype is not None else None)
    dataset.set_preprocessor(
        PrismaticPreprocessor(
            model.vision_backbone.vision_transform,  # type: ignore
            model.get_prompt_builder(),
        )
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
    data: list = []
    for i, batch in enumerate(
        tqdm(dataloader, desc="Generating", total=num_eval_steps)
    ):
        if num_eval_steps is not None and i == num_eval_steps:
            break
        gathered_objects: list[list] = []
        for column in dataset.columns:
            gathered_objects.append(gather_object(batch[column]))
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
            for i in range(len(gathered_objects)):
                gathered_objects[i] = gathered_objects[i][
                    : accelerator.gradient_state.remainder
                ]
            gathered_gen_texts = gathered_gen_texts[
                : accelerator.gradient_state.remainder
            ]
        data.extend(zip(*gathered_objects, gathered_gen_texts))

    columns = dataset.columns + ["generated"]
    if out_file_name is not None and accelerator.is_main_process:
        with open(out_file_name, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            writer.writerows(data)

    if wandb_project is not None and accelerator.is_main_process:
        accelerator.get_tracker("wandb").log_table(
            "generated", columns=columns, data=data
        )

    accelerator.end_training()


if __name__ == "__main__":
    CLI(run, as_positional=False)
