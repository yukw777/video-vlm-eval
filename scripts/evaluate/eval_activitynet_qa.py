import ast
import csv
import json
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


@dataclass
class OpenAIClient:
    client: OpenAI

    @retry(wait=wait_random_exponential(), stop=stop_after_attempt(100))
    def annotate(self, question: str, answer: str, pred: str) -> dict:
        # Compute the correctness score
        chat_completion = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
                    "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:"
                    "------"
                    "##INSTRUCTIONS: "
                    "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
                    "- Consider synonyms or paraphrases as valid matches.\n"
                    "- Evaluate the correctness of the prediction compared to the answer.",
                },
                {
                    "role": "user",
                    "content": "Please evaluate the following video-based question-answer pair:\n\n"
                    f"Question: {question}\n"
                    f"Correct Answer: {answer}\n"
                    f"Predicted Answer: {pred}\n\n"
                    "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. "
                    "Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is  a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING."
                    "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                    "For example, your response should look like this: {'pred': 'yes', 'score': 4.8}.",
                },
            ],
        )
        # Convert response to a Python dictionary.
        response_message = chat_completion.choices[0].message.content
        assert isinstance(response_message, str)
        return ast.literal_eval(response_message)


def run(
    openai_api_key: str, pred_path: str, gt_file_answer: str, out_path: str | None
) -> None:
    wandb.init()
    with open(pred_path, newline="") as f:
        reader = csv.DictReader(f)
        preds = {p["question_id"]: p for p in reader}
    with open(gt_file_answer) as f:
        answers = {a["question_id"]: a for a in json.load(f)}

    client = OpenAIClient(OpenAI(api_key=openai_api_key))

    data: list[list] = []
    anns: list[dict] = []
    with ThreadPoolExecutor() as executor:
        future_to_q_id = {
            executor.submit(
                client.annotate,
                preds[q_id]["question"],
                answers[q_id]["answer"],
                preds[q_id]["generated"],
            ): q_id
            for q_id in preds
        }
        for future in tqdm(as_completed(future_to_q_id), total=len(preds)):
            q_id = future_to_q_id[future]
            ann = future.result()
            anns.append(ann)
            data.append(
                [
                    preds[q_id]["video_name"],
                    preds[q_id]["question_id"],
                    preds[q_id]["question"],
                    answers[q_id]["answer"],
                    preds[q_id]["generated"],
                    ann["pred"],
                    ann["score"],
                ]
            )
    columns = [
        "video_name",
        "question_id",
        "question",
        "answer",
        "generated",
        "chatgpt_pred",
        "chatgpt_score",
    ]
    if out_path is not None:
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            writer.writerows(data)
    table = wandb.Table(columns=columns, data=data)

    # Calculate average score and accuracy
    score_sum = 0
    count = 0
    yes_count = 0
    no_count = 0
    for ann in anns:
        # Computing score
        count += 1
        score = int(ann["score"])
        score_sum += score

        # Computing accuracy
        pred = ann["pred"]
        if "yes" in pred.lower():
            yes_count += 1
        elif "no" in pred.lower():
            no_count += 1

    average_score = score_sum / count
    accuracy = yes_count / (yes_count + no_count)
    log_dict = {
        "eval_results": table,
        "yes_count": yes_count,
        "no_count": no_count,
        "accuracy": accuracy,
        "average_score": average_score,
    }
    pprint(log_dict)
    wandb.log(log_dict)


if __name__ == "__main__":
    CLI(run, as_positional=False)
