import wandb
import kaggle
import re
import csv
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
    print("Submitting the following inference runs to Kaggle:")
    pprint([run.name for run in inference_runs])
    print("===========================================")
    for inference_run in inference_runs:
        table = inference_run.logged_artifacts()[0]["inference"]
        df = table.get_dataframe()
        submission_file_name = f"kaggle-{inference_run.name}.csv"
        with open(submission_file_name, newline="", mode="w") as f_out:
            writer = csv.DictWriter(f_out, ("q_uid", "answer"))
            writer.writeheader()
            for row in df.to_dict(orient="records"):
                pred = str(row["pred"])
                if not ("0" <= pred <= "4"):
                    # it's an invalid answer so replace it ith "5" so it'd be marked incorrect.
                    pred = "5"
                writer.writerow({"q_uid": row["q_uid"], "answer": pred})
        kaggle.api.competition_submit(
            submission_file_name, inference_run.name, "egoschema-public"
        )


if __name__ == "__main__":
    CLI(main, as_positional=False)
