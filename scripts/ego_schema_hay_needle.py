import json
import random


def main(
    question_file: str, needle_haystack_mapping_file: str, haystack_size: int
) -> None:
    with open(question_file) as f:
        questions = json.load(f)
    q_uids = set(q["q_uid"] for q in questions)
    needle_haystack_mapping = {
        q_uid: random.sample(
            [q_uid]
            + random.sample([uid for uid in q_uids if uid != q_uid], haystack_size),
            haystack_size + 1,
        )
        for q_uid in q_uids
    }
    with open(needle_haystack_mapping_file, mode="w") as f:
        json.dump(needle_haystack_mapping, f)


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(main, as_positional=False)
