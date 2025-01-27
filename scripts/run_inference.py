import json
import csv
import os

from accelerate import Accelerator
from accelerate.utils import gather_object, set_seed, tqdm
from jsonargparse import CLI
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Subset

from video_vlm_eval import Dataset, Model

# Disable Tokenizers Parallelism to Play Nice w/ PyTorch Multiprocessing DataLoaders
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def run(
    model: Model,
    dataset: Dataset,
    per_device_batch_size: int = 2,
    num_dataloader_workers: int = 4,
    start_idx: int | None = None,
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
        accelerator.init_trackers(wandb_project)
    else:
        accelerator = Accelerator()

    dataset.set_preprocessor(model.preprocess)
    dataset_columns = dataset.columns
    if start_idx is not None:
        dataset = Subset(dataset, range(start_idx, len(dataset)))  # type: ignore
    model, dataloader = accelerator.prepare(
        model,
        DataLoader(
            dataset,
            batch_size=per_device_batch_size,
            num_workers=num_dataloader_workers,
            pin_memory=True,
            collate_fn=model.collate_fn,
        ),
    )

    model = model.module if isinstance(model, DistributedDataParallel) else model
    data: list = []
    for i, batch in enumerate(tqdm(dataloader, desc="Inference", total=num_eval_steps)):
        if num_eval_steps is not None and i == num_eval_steps:
            break
        gathered_objects: list[list] = []
        for column in dataset_columns:
            batched_obj = batch[column]
            if not isinstance(batched_obj[0], str):
                # if the batched element is not a string, e.g., list of strings,
                # dump it as a json
                batched_obj = [json.dumps(obj) for obj in batched_obj]
            gathered_objects.append(gather_object(batched_obj))
        task_results = model.perform(batch, **gen_config)
        gathered_task_results = gather_object(task_results)
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
            gathered_task_results = gathered_task_results[
                : accelerator.gradient_state.remainder
            ]
        data.extend(
            zip(
                *gathered_objects,
                *[
                    [result[key] for result in gathered_task_results]
                    for key in model.result_keys
                ],
                strict=True,
            )
        )

    out_columns = dataset_columns + model.result_keys
    if out_file_name is not None and accelerator.is_main_process:
        with open(out_file_name, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(out_columns)
            writer.writerows(data)

    if wandb_project is not None and accelerator.is_main_process:
        accelerator.get_tracker("wandb").log_table(
            "inference", columns=out_columns, data=data
        )

    accelerator.end_training()


if __name__ == "__main__":
    CLI(run, as_positional=False)
