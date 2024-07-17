import csv
import json


def main(pred_file: str, subset_answer_file: str, out_file: str) -> None:
    with open(subset_answer_file) as f:
        answers = json.load(f)
    with open(pred_file) as f:
        reader = csv.DictReader(f)
        preds = list(reader)

    with open(out_file, "w") as f:
        writer = csv.DictWriter(f, fieldnames=preds[0].keys())
        writer.writeheader()
        for pred in preds:
            pred["answer"] = str(answers.get(pred["q_uid"], ""))
            writer.writerow(pred)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("pred_file")
    parser.add_argument("subset_answer_file")
    parser.add_argument("out_file")
    args = parser.parse_args()

    main(args.pred_file, args.subset_answer_file, args.out_file)
