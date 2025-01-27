import wandb
import json
import re
from pprint import pprint
from jsonargparse import CLI


def main(
    inference_entity: str, inference_project: str, inference_run_name_regex: str
) -> None:
    wandb_api = wandb.Api()
    pattern = re.compile(inference_run_name_regex)
    inference_runs = [
        run
        for run in wandb_api.runs(f"{inference_entity}/{inference_project}")
        if pattern.search(run.name)
    ]
    print("Preparing submissions for the following inference runs:")
    pprint([run.name for run in inference_runs])
    print("===========================================")
    for inference_run in inference_runs:
        table = inference_run.logged_artifacts()[0]["inference"]
        df = table.get_dataframe()
        preds = []
        for row in df.to_dict(orient="records"):
            preds.append(
                {
                    "question_id": row["question_id"],
                    "question_type": row["question_type"],
                    "option": row["pred"],
                }
            )
        submission_file_name = f"{inference_run.name}.json"
        with open(submission_file_name, mode="w") as f_out:
            json.dump(preds, f_out)


if __name__ == "__main__":
    CLI(main, as_positional=False)
