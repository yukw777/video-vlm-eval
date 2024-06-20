import csv
import json
import wandb
from jsonargparse import CLI
from video_vlm_eval import Task
from pprint import pprint


def run(task: Task, pred_path: str, out_path: str | None) -> None:
    wandb.init()
    with open(pred_path, newline="") as f:
        reader = csv.DictReader(f)
        anns = [ann for ann in reader]
    metrics = task.calculate_metrics(anns)
    if out_path is not None:
        with open(out_path, "w") as f:
            json.dump(metrics, f)
    pprint(metrics)
    wandb.log(metrics)


if __name__ == "__main__":
    CLI(run, as_positional=False)
