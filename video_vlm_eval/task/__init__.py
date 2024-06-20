import abc


class Task(abc.ABC):
    question_key = "question"
    answer_key = "answer"
    pred_key = "pred"

    @abc.abstractmethod
    def calculate_metrics(self, preds: list[dict]) -> dict: ...


class ZeroShotQA(Task):
    pred_key = "generated"


class MultipleChoice(Task):
    def calculate_metrics(self, anns: list[dict]) -> dict:
        match_count = sum(ann[self.answer_key] == ann[self.pred_key] for ann in anns)
        return {"accuracy": match_count / len(anns)}
