import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pprint import pprint

import wandb
from jsonargparse import CLI
from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from tqdm import tqdm

from video_vlm_eval import Dataset, OpenAIEvalTask


@dataclass
class OpenAIClient:
    task: OpenAIEvalTask
    client: OpenAI

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10))
    def annotate(self, request: dict) -> dict:
        # Compute the correctness score
        chat_completion = self.client.chat.completions.create(**request)
        return self.task.parse_openai_response(
            chat_completion.choices[0].message.content
        )


def run(
    task: OpenAIEvalTask,
    dataset: Dataset,
    openai_api_key: str,
    pred_path: str,
    out_path: str | None,
) -> None:
    wandb.init()
    with open(pred_path, newline="") as f:
        reader = csv.DictReader(f)
        preds = {p[dataset.id_key]: p for p in reader}

    client = OpenAIClient(task, OpenAI(api_key=openai_api_key))

    data: list[list] = []
    anns: list[dict] = []
    with ThreadPoolExecutor() as executor:
        future_to_q_id = {
            executor.submit(
                client.annotate,
                task.get_openai_request(dataset.get_by_id(q_id), preds[q_id]),
            ): q_id
            for q_id in preds
        }
        for future in tqdm(as_completed(future_to_q_id), total=len(preds)):
            q_id = future_to_q_id[future]
            ann = future.result()
            anns.append(
                {
                    **{
                        k: v
                        for k, v in dataset.get_by_id(q_id).items()
                        if k in dataset.columns
                    },
                    **{k: v for k, v in preds[q_id].items() if k in task.pred_keys},
                    **ann,
                }
            )
            data.append(
                [dataset.get_by_id(q_id)[c] for c in dataset.columns]
                + [preds[q_id][key] for key in task.pred_keys]
                + [ann[key] for key in task.ann_keys]
            )
    columns = dataset.columns + task.pred_keys + task.ann_keys
    if out_path is not None:
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            writer.writerows(data)
    table = wandb.Table(columns=columns, data=data)
    metrics = task.calculate_metrics(anns)
    log_dict = {"eval_results": table, **metrics}
    pprint(log_dict)
    wandb.log(log_dict)


if __name__ == "__main__":
    CLI(run, as_positional=False)
