import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pprint import pprint

import wandb
from jsonargparse import CLI
from openai import OpenAI
from tenacity import retry, wait_random_exponential
from tqdm import tqdm

from video_vlm_eval import Dataset, OpenAIEvalTask


@dataclass
class OpenAIClient:
    task: OpenAIEvalTask
    client: OpenAI

    @retry(wait=wait_random_exponential(min=1, max=60))
    def annotate(self, request: dict) -> dict:
        # Compute the correctness score
        chat_completion = self.client.chat.completions.create(**request)
        return self.task.parse_openai_response(
            chat_completion.choices[0].message.content
        )


def main(
    task: OpenAIEvalTask,
    dataset: Dataset,
    openai_api_key: str,
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
        preds = df.to_dict(orient="records")
        eval_results = run_eval(task, dataset, openai_api_key, preds)
        with wandb.init(
            entity=eval_entity, project=eval_project, name=inference_run.name
        ) as eval_run:
            eval_run.log(eval_results)


def run_eval(
    task: OpenAIEvalTask, dataset: Dataset, openai_api_key: str, preds: dict
) -> dict:
    client = OpenAIClient(task, OpenAI(api_key=openai_api_key))

    data: list[list] = []
    anns: list[dict] = []
    with ThreadPoolExecutor() as executor:
        future_to_pred = {
            executor.submit(
                client.annotate,
                task.get_openai_request(dataset.get_by_id(pred[dataset.id_key]), pred),
            ): pred
            for pred in preds
        }
        for future in tqdm(as_completed(future_to_pred), total=len(preds)):
            pred = future_to_pred[future]
            ann = future.result()
            anns.append(
                {
                    **{
                        k: v
                        for k, v in dataset.get_by_id(pred[dataset.id_key]).items()
                        if k in dataset.columns
                    },
                    **{k: v for k, v in pred.items() if k in task.pred_keys},
                    **ann,
                }
            )
            data.append(
                [dataset.get_by_id(pred[dataset.id_key])[c] for c in dataset.columns]
                + [pred[key] for key in task.pred_keys]
                + [ann[key] for key in task.ann_keys]
            )
    columns = dataset.columns + task.pred_keys + task.ann_keys
    table = wandb.Table(columns=columns, data=data)
    metrics = task.calculate_metrics(anns)
    log_dict = {"eval_results": table, **metrics}
    pprint(log_dict)
    return log_dict


if __name__ == "__main__":
    CLI(main, as_positional=False)
