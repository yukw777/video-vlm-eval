import csv


def main(pred_file: str, submission_file: str) -> None:
    with (
        open(pred_file, newline="") as f_in,
        open(submission_file, newline="", mode="w") as f_out,
    ):
        reader = csv.DictReader(f_in)
        writer = csv.DictWriter(f_out, ("q_uid", "answer"))
        writer.writeheader()
        for row in reader:
            if not ("0" <= row["pred"] <= "4"):
                # it's an invalid answer so replace it ith "5" so it'd be marked incorrect.
                row["pred"] = "5"
            writer.writerow({"q_uid": row["q_uid"], "answer": row["pred"]})


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("pred_file")
    parser.add_argument("submission_file")
    args = parser.parse_args()

    main(args.pred_file, args.submission_file)
