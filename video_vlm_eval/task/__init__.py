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
        match_count = sum(
            str(ann[self.answer_key]) == str(ann[self.pred_key]) for ann in anns
        )
        return {"accuracy": match_count / len(anns)}


class OpenAIEvalTask(abc.ABC):
    @abc.abstractmethod
    def get_openai_request(self, datapoint: dict, pred: dict) -> dict: ...

    @abc.abstractmethod
    def parse_openai_response(self, response_message: str) -> dict: ...

    @abc.abstractmethod
    def calculate_metrics(self, anns: list[dict]) -> dict: ...

    @property
    @abc.abstractmethod
    def pred_keys(self) -> list[str]: ...

    @property
    @abc.abstractmethod
    def ann_keys(self) -> list[str]: ...
