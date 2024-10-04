import wandb
import re
from jsonargparse import CLI
from video_vlm_eval import Task
from pprint import pprint


def main(
    task: Task,
    inference_entity: str,
    inference_project: str,
    inference_run_name_regex: str,
    eval_project: str,
    eval_entity: str | None = None,
) -> None:
    if eval_entity is None:
        eval_entity = inference_entity
    wandb_api = wandb.Api()
    pattern = re.compile(inference_run_name_regex)
    inference_runs = [
        run
        for run in wandb_api.runs(f"{inference_entity}/{inference_project}")
        if pattern.search(run.name)
    ]
    print("Running evaluation for the following inference runs:")
    pprint([run.name for run in inference_runs])
    print("===========================================")
    for inference_run in inference_runs:
        table = inference_run.logged_artifacts()[0]["inference"]
        df = table.get_dataframe()
        anns = df.to_dict(orient="records")
        metrics = task.calculate_metrics(anns)
        print(f"==== Metrics for {inference_run.name} ====")
        pprint(metrics)
        print("===========================================")
        with wandb.init(
            entity=eval_entity, project=eval_project, name=inference_run.name
        ) as eval_run:
            eval_run.log(metrics)


if __name__ == "__main__":
    CLI(main, as_positional=False)
