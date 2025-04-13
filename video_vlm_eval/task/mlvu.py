from video_vlm_eval.task import ZeroShotQA, OpenAIEvalTask, MultipleChoice


class MLVUSSCTask(ZeroShotQA, OpenAIEvalTask):
    # Based on https://github.com/JUNJIE99/MLVU/blob/ee8ac0984f7cdfc53730a0aecde5c7ce457570fd/evaluation/generation_evaluation/evaluate_ssc.py
    # and https://github.com/JUNJIE99/MLVU/blob/ee8ac0984f7cdfc53730a0aecde5c7ce457570fd/evaluation/generation_evaluation/calculate.py.

    SYSTEM_MSG = (
        "##TASK DESCRIPTION:\n"
        "You are required to evaluate a respondent's answer based on a provided question, some scoring points, and the respondent's answer. You should provide two scores. The first is the accuracy score, which should range from 1 to 5. The second is the relevance score, which should also range from 1 to 5. Below are the criteria for each scoring category.\n"
        "##ACCURACY Scoring Criteria: \n"
        "Evaluate the respondent's answer against specific scoring points as follows:\n"
        "Score 1: The response completely misses the scoring point.\n"
        "Score 3: The response mentions content related to the scoring point but is not entirely correct.\n"
        "Score 5: The response accurately addresses the scoring point.\n"
        "Calculate the average score across all scoring points to determine the final accuracy score.\n"
        "##RELEVANCE Scoring Criteria:\n"
        "Assess how the respondent's answer relates to the original question:\n"
        "Score 1: The response is completely off-topic from the question.\n"
        "Score 2: The response is partially related to the question but contains a significant amount of irrelevant content.\n"
        "Score 3: The response primarily addresses the question, but the respondent seems uncertain about their own answer.\n"
        "Score 4: The response mostly addresses the question and the respondent appears confident in their answer.\n"
        "Score 5: The response is fully focused on addressing the question with no irrelevant content and demonstrates complete certainty.\n"
        "----\n"
        "##INSTRUCTION:\n"
        "1. Evaluate Accuracy: First, assess and score each scoring point based on the respondent's answer. Calculate the average of these scores to establish the final accuracy score. Provide a detailed rationale before assigning your score.\n"
        "2. Evaluate RELEVANCE: Assess the relevance of the respondent's answer to the question. Note that when evaluating relevance, the correctness of the answer is not considered; focus solely on how relevant the answer is to the question. Provide a comprehensive rationale before assigning your score.\n"
        "3. Output Scores in JSON Format: Present the scores in JSON format as follows:\n"
        "{'score_accuracy': score_acc, 'score_relevance': score_rele, 'total_score': score_acc + score_rele}"
    )
    USER_MSG = (
        "Please score the respondent's answer according to the steps in the Instructions. You must end with a JSON dict to store the scores.\n"
        "Question: {question}\n"
        "Scoring Points: {scoring_points}\n"
        "Respondent's Answer: {pred}"
    )

    def get_openai_request(self, datapoint: dict, pred: dict) -> dict:
        return {
            "temperature": 0,
            "model": "gpt-4-turbo",
            "messages": [
                {"role": "system", "content": self.SYSTEM_MSG},
                {
                    "role": "user",
                    "content": self.USER_MSG.format(
                        question=datapoint[self.question_key],
                        scoring_points=datapoint["scoring_points"],
                        pred=pred[self.pred_key],
                    ),
                },
            ],
        }

    def parse_openai_response(self, response_message: str) -> dict:
        scores = {}

        for key in self.ann_keys:
            # Find the index where each key starts
            start_index = response_message.find(key)
            if start_index == -1:
                # Assign 0 if key is not found
                scores[key] = 0.0
                continue

            # Find the start of the number which is after the colon and space
            start_number_index = response_message.find(":", start_index) + 2
            end_number_index = response_message.find(
                ",", start_number_index
            )  # Assuming the number ends before a comma

            # Extract and convert the number to float
            score = float(response_message[start_number_index:end_number_index])
            scores[key] = score

        return scores

    @property
    def pred_keys(self) -> list[str]:
        return [self.pred_key]

    @property
    def ann_keys(self) -> list[str]:
        return ["score_accuracy", "score_relevance"]

    def calculate_metrics(self, anns: list[dict]) -> dict:
        # Calculate average accuracy, relevance and total score
        accu = sum(ann[self.ann_keys[0]] for ann in anns) / len(anns)
        rele = sum(ann[self.ann_keys[1]] for ann in anns) / len(anns)
        total = accu + rele

        return {"accuracy": accu, "relevance": rele, "total": total}


class MLVUSummaryTask(ZeroShotQA, OpenAIEvalTask):
    # Based on https://github.com/JUNJIE99/MLVU/blob/ee8ac0984f7cdfc53730a0aecde5c7ce457570fd/evaluation/generation_evaluation/evaluate_summary.py
    # and https://github.com/JUNJIE99/MLVU/blob/ee8ac0984f7cdfc53730a0aecde5c7ce457570fd/evaluation/generation_evaluation/calculate_sum.py.

    SYSTEM_MSG = (
        "##TASK DESCRIPTION:\n"
        "You are required to evaluate the performance of the respondent in the video summarization task based on the standard answer and the respondent's answer. You should provide two scores. The first is the COMPLETENESS score, which should range from 1 to 5. The second is the RELIABILITY score, which should also range from 1 to 5. Below are the criteria for each scoring category:\n"
        "##COMPLETENESS Scoring Criteria:\n"
        "The completeness score focuses on whether the summary covers all key points and main information from the video.\n"
        "Score 1: The summary hardly covers any of the main content or key points of the video.\n"
        "Score 2: The summary covers some of the main content and key points but misses many.\n"
        "Score 3: The summary covers most of the main content and key points.\n"
        "Score 4: The summary is very comprehensive, covering most to nearly all of the main content and key points.\n"
        "Score 5: The summary completely covers all the main content and key points of the video.\n"
        "##RELIABILITY Scoring Criteria:\n"
        "The reliability score evaluates the correctness and clarity of the video summary. It checks for factual errors, misleading statements, and contradictions with the video content. If the respondent's answer includes details that are not present in the standard answer, as long as these details do not conflict with the correct answer and are reasonable, points should not be deducted.\n"
        "Score 1: Contains multiple factual errors and contradictions; presentation is confusing.\n"
        "Score 2: Includes several errors and some contradictions; needs clearer presentation.\n"
        "Score 3: Generally accurate with minor errors; minimal contradictions; reasonably clear presentation.\n"
        "Score 4: Very accurate with negligible inaccuracies; no contradictions; clear and fluent presentation.\n"
        "Score 5: Completely accurate with no errors or contradictions; presentation is clear and easy to understand.\n"
        "----\n"
        "##INSTRUCTION:\n"
        "1. Evaluate COMPLETENESS: First, analyze the respondent's answer according to the scoring criteria, then provide an integer score between 1 and 5 based on sufficient evidence.\n"
        "2. Evaluate RELIABILITY: First, analyze the respondent's answer according to the scoring criteria, then provide an integer score between 1 and 5 based on sufficient evidence.\n"
        "3. Output Scores in JSON Format: Present the scores in JSON format as follows:\n"
        "{'score_completeness': score_comp, 'score_reliability': score_reli, 'total_score': score_comp + score_reli}\n"
    )
    USER_MSG = (
        "Please score the respondent's answer according to the steps in the Instructions. You must end with a JSON dict to store the scores.\n"
        "Standard Answer: {answer}\n"
        "Respondent's Answer: {pred}\n"
    )

    def get_openai_request(self, datapoint: dict, pred: dict) -> dict:
        return {
            "temperature": 0,
            "model": "gpt-4-turbo",
            "messages": [
                {"role": "system", "content": self.SYSTEM_MSG},
                {
                    "role": "user",
                    "content": self.USER_MSG.format(
                        answer=datapoint[self.answer_key], pred=pred[self.pred_key]
                    ),
                },
            ],
        }

    def parse_openai_response(self, response_message: str) -> dict:
        scores = {}

        for key in self.ann_keys:
            # Find the index where each key starts
            start_index = response_message.find(key)
            if start_index == -1:
                # Assign 0 if key is not found
                scores[key] = 0.0
                continue

            # Find the start of the number which is after the colon and space
            start_number_index = response_message.find(":", start_index) + 2
            end_number_index = response_message.find(
                ",", start_number_index
            )  # Assuming the number ends before a comma

            # Extract and convert the number to float
            score = float(response_message[start_number_index:end_number_index])
            scores[key] = score

        return scores

    @property
    def pred_keys(self) -> list[str]:
        return [self.pred_key]

    @property
    def ann_keys(self) -> list[str]:
        return ["score_completeness", "score_reliability"]

    def calculate_metrics(self, anns: list[dict]) -> dict:
        # Calculate average completeness, reliability and total score
        comp = sum(ann[self.ann_keys[0]] for ann in anns) / len(anns)
        reli = sum(ann[self.ann_keys[1]] for ann in anns) / len(anns)
        total = comp + reli

        return {"completeness": comp, "reliability": reli, "total": total}


class MLVUMultiplechoice(MultipleChoice):
    def calculate_metrics(self, anns: list[dict]) -> dict:
        # calculate accuracy for each task type
        task_types = {ann["task_type"] for ann in anns}
        metrics = {}
        for task_type in task_types:
            task_anns = [ann for ann in anns if ann["task_type"] == task_type]
            acc = super().calculate_metrics(task_anns)["accuracy"]
            metrics[f"{task_type}_accuracy"] = acc
        metrics["avg"] = sum(metrics.values()) / len(metrics)
        return metrics
