import abc
import ast
from video_vlm_eval.task import ZeroShotQA


class VideoChatGPTTask(abc.ABC):
    @abc.abstractmethod
    def get_openai_request(self, datapoint: dict, pred: dict) -> dict: ...

    @abc.abstractmethod
    def parse_openai_response(self, response_message: str) -> dict: ...

    @property
    @abc.abstractmethod
    def pred_keys(self) -> list[str]: ...

    @property
    @abc.abstractmethod
    def ann_keys(self) -> list[str]: ...


class VideoChatGPTZeroShotQATask(ZeroShotQA, VideoChatGPTTask):
    SYSTEM_MSG = (
        "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
        + "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:"
        + "------"
        + "##INSTRUCTIONS: "
        + "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
        + "- Consider synonyms or paraphrases as valid matches.\n"
        + "- Evaluate the correctness of the prediction compared to the answer."
    )
    USER_MSG = (
        "Please evaluate the following video-based question-answer pair:\n\n"
        + "Question: {question}\n"
        + "Correct Answer: {answer}\n"
        + "Predicted Answer: {pred}\n\n"
        + "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. "
        + "Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is  a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING."
        + "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
        + "For example, your response should look like this: {{'pred': 'yes', 'score': 4.8}}."
    )

    def get_openai_request(self, datapoint: dict, pred: dict) -> dict:
        return {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": self.SYSTEM_MSG},
                {
                    "role": "user",
                    "content": self.USER_MSG.format(
                        question=datapoint[self.question_key],
                        answer=datapoint[self.answer_key],
                        pred=pred[self.pred_key],
                    ),
                },
            ],
        }

    def parse_openai_response(self, response_message: str) -> dict:
        parsed = ast.literal_eval(response_message)
        return {"chatgpt_pred": parsed["pred"], "chatgpt_score": parsed["score"]}

    @property
    def pred_keys(self) -> list[str]:
        return [self.pred_key]

    @property
    def ann_keys(self) -> list[str]:
        return ["chatgpt_pred", "chatgpt_score"]

    def calculate_metrics(self, anns: list[dict]) -> dict:
        # Calculate average score and accuracy
        score_sum = 0
        count = 0
        yes_count = 0
        no_count = 0
        for ann in anns:
            # Computing score
            count += 1
            score = int(ann["chatgpt_score"])
            score_sum += score

            # Computing accuracy
            pred = ann["chatgpt_pred"]
            if "yes" in pred.lower():
                yes_count += 1
            elif "no" in pred.lower():
                no_count += 1

        average_score = score_sum / count
        accuracy = yes_count / (yes_count + no_count)

        return {
            "yes_count": yes_count,
            "no_count": no_count,
            "accuracy": accuracy,
            "average_score": average_score,
        }


class VideoChatGPTGeneralTask(VideoChatGPTTask):
    SYSTEM_MSG: str
    USER_MSG: str

    question_keys = ["question"]
    answer_key = "answer"
    pred_keys = ["generated"]

    def get_openai_request(self, datapoint: dict, pred: dict) -> dict:
        return {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": self.SYSTEM_MSG},
                {
                    "role": "user",
                    "content": self.USER_MSG.format(
                        question=datapoint[self.question_keys[0]],
                        answer=datapoint[self.answer_key],
                        pred=pred[self.pred_keys[0]],
                    ),
                },
            ],
        }

    def parse_openai_response(self, response_message: str) -> dict:
        parsed = ast.literal_eval(response_message)
        return {"chatgpt_score": parsed["score"]}

    @property
    def ann_keys(self) -> list[str]:
        return ["chatgpt_score"]

    def calculate_metrics(self, anns: list[dict]) -> dict:
        # Calculate average score
        score_sum = 0
        count = 0
        for ann in anns:
            count += 1
            score = int(ann["chatgpt_score"])
            score_sum += score
        average_score = score_sum / count
        return {"average_score": average_score}


class VideoChatGPTCorrectnessTask(VideoChatGPTGeneralTask):
    SYSTEM_MSG = (
        "You are an intelligent chatbot designed for evaluating the factual accuracy of generative outputs for video-based question-answer pairs. "
        + "Your task is to compare the predicted answer with the correct answer and determine if they are factually consistent. Here's how you can accomplish the task:"
        + "------"
        + "##INSTRUCTIONS: "
        + "- Focus on the factual consistency between the predicted answer and the correct answer. The predicted answer should not contain any misinterpretations or misinformation.\n"
        + "- The predicted answer must be factually accurate and align with the video content.\n"
        + "- Consider synonyms or paraphrases as valid matches.\n"
        + "- Evaluate the factual accuracy of the prediction compared to the answer."
    )
    USER_MSG = (
        "Please evaluate the following video-based question-answer pair:\n\n"
        + "Question: {question}\n"
        + "Correct Answer: {answer}\n"
        + "Predicted Answer: {pred}\n\n"
        + "Provide your evaluation only as a factual accuracy score where the factual accuracy score is an integer value between 0 and 5, with 5 indicating the highest level of factual consistency. "
        + "Please generate the response in the form of a Python dictionary string with keys 'score', where its value is the factual accuracy score in INTEGER, not STRING."
        + "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
        + "For example, your response should look like this: {{''score': 4.8}}."
    )


class VideoChatGPTDetailedOrientationTask(VideoChatGPTGeneralTask):
    SYSTEM_MSG = (
        "You are an intelligent chatbot designed for evaluating the detail orientation of generative outputs for video-based question-answer pairs. "
        + "Your task is to compare the predicted answer with the correct answer and determine its level of detail, considering both completeness and specificity. Here's how you can accomplish the task:"
        + "------"
        + "##INSTRUCTIONS: "
        + "- Check if the predicted answer covers all major points from the video. The response should not leave out any key aspects.\n"
        + "- Evaluate whether the predicted answer includes specific details rather than just generic points. It should provide comprehensive information that is tied to specific elements of the video.\n"
        + "- Consider synonyms or paraphrases as valid matches.\n"
        + "- Provide a single evaluation score that reflects the level of detail orientation of the prediction, considering both completeness and specificity."
    )
    USER_MSG = (
        "Please evaluate the following video-based question-answer pair:\n\n"
        + "Question: {question}\n"
        + "Correct Answer: {answer}\n"
        + "Predicted Answer: {pred}\n\n"
        + "Provide your evaluation only as a detail orientation score where the detail orientation score is an integer value between 0 and 5, with 5 indicating the highest level of detail orientation. "
        + "Please generate the response in the form of a Python dictionary string with keys 'score', where its value is the detail orientation score in INTEGER, not STRING."
        + "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
        + "For example, your response should look like this: {{''score': 4.8}}."
    )


class VideoChatGPTContextTask(VideoChatGPTGeneralTask):
    SYSTEM_MSG = (
        "You are an intelligent chatbot designed for evaluating the contextual understanding of generative outputs for video-based question-answer pairs. "
        + "Your task is to compare the predicted answer with the correct answer and determine if the generated response aligns with the overall context of the video content. Here's how you can accomplish the task:"
        + "------"
        + "##INSTRUCTIONS: "
        + "- Evaluate whether the predicted answer aligns with the overall context of the video content. It should not provide information that is out of context or misaligned.\n"
        + "- The predicted answer must capture the main themes and sentiments of the video.\n"
        + "- Consider synonyms or paraphrases as valid matches.\n"
        + "- Provide your evaluation of the contextual understanding of the prediction compared to the answer."
    )
    USER_MSG = (
        "Please evaluate the following video-based question-answer pair:\n\n"
        + "Question: {question}\n"
        + "Correct Answer: {answer}\n"
        + "Predicted Answer: {pred}\n\n"
        + "Provide your evaluation only as a contextual understanding score where the contextual understanding score is an integer value between 0 and 5, with 5 indicating the highest level of contextual understanding. "
        + "Please generate the response in the form of a Python dictionary string with keys 'score', where its value is contextual understanding score in INTEGER, not STRING."
        + "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
        + "For example, your response should look like this: {{''score': 4.8}}."
    )


class VideoChatGPTTemporalTask(VideoChatGPTGeneralTask):
    SYSTEM_MSG = (
        "You are an intelligent chatbot designed for evaluating the temporal understanding of generative outputs for video-based question-answer pairs. "
        + "Your task is to compare the predicted answer with the correct answer and determine if they correctly reflect the temporal sequence of events in the video content. Here's how you can accomplish the task:"
        + "------"
        + "##INSTRUCTIONS: "
        + "- Focus on the temporal consistency between the predicted answer and the correct answer. The predicted answer should correctly reflect the sequence of events or details as they are presented in the video content.\n"
        + "- Consider synonyms or paraphrases as valid matches, but only if the temporal order is maintained.\n"
        + "- Evaluate the temporal accuracy of the prediction compared to the answer."
    )
    USER_MSG = (
        "Please evaluate the following video-based question-answer pair:\n\n"
        + "Question: {question}\n"
        + "Correct Answer: {answer}\n"
        + "Predicted Answer: {pred}\n\n"
        + "Provide your evaluation only as a temporal accuracy score where the temporal accuracy score is an integer value between 0 and 5, with 5 indicating the highest level of temporal consistency. "
        + "Please generate the response in the form of a Python dictionary string with keys 'score', where its value is the temporal accuracy score in INTEGER, not STRING."
        + "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
        + "For example, your response should look like this: {{''score': 4.8}}."
    )


class VideoChatGPTConsistencyTask(VideoChatGPTGeneralTask):
    question_keys = ["Q1", "Q2"]
    answer_key = "A"
    pred_keys = ["P1", "P2"]

    SYSTEM_MSG = (
        "You are an intelligent chatbot designed for evaluating the consistency of generative outputs for similar video-based question-answer pairs. "
        + "You will be given two very similar questions, a common answer common to both the questions and predicted answers for the two questions ."
        + "Your task is to compare the predicted answers for two very similar question, with a common correct answer and determine if they are consistent. Here's how you can accomplish the task:"
        + "------"
        + "##INSTRUCTIONS: "
        + "- Focus on the consistency between the two predicted answers and the correct answer. Both predicted answers should correspond to the correct answer and to each other, and should not contain any contradictions or significant differences in the conveyed information.\n"
        + "- Both predicted answers must be consistent with each other and the correct answer, in terms of the information they provide about the video content.\n"
        + "- Consider synonyms or paraphrases as valid matches, but only if they maintain the consistency in the conveyed information.\n"
        + "- Evaluate the consistency of the two predicted answers compared to the correct answer."
    )
    USER_MSG = (
        "Please evaluate the following video-based question-answer pair:\n\n"
        + "Question 1: {question1}\n"
        + "Question 2: {question2}\n"
        + "Correct Answer: {answer}\n"
        + "Predicted Answer to Question 1: {pred1}\n"
        + "Predicted Answer to Question 2: {pred2}\n\n"
        + "Provide your evaluation only as a consistency score where the consistency score is an integer value between 0 and 5, with 5 indicating the highest level of consistency. "
        + "Please generate the response in the form of a Python dictionary string with keys 'score', where its value is the consistency score in INTEGER, not STRING."
        + "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
        + "For example, your response should look like this: {{''score': 4.8}}."
    )

    def get_openai_request(self, datapoint: dict, pred: dict) -> dict:
        return {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": self.SYSTEM_MSG},
                {
                    "role": "user",
                    "content": self.USER_MSG.format(
                        question1=datapoint[self.question_keys[0]],
                        question2=datapoint[self.question_keys[1]],
                        answer=datapoint[self.answer_key],
                        pred1=pred[self.pred_keys[0]],
                        pred2=pred[self.pred_keys[1]],
                    ),
                },
            ],
        }
