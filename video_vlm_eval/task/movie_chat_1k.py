from video_vlm_eval.task.video_chatgpt import VideoChatGPTZeroShotQATask


class MovieChat1KTask(VideoChatGPTZeroShotQATask):
    def calculate_metrics(self, anns: list[dict]) -> dict:
        g_anns = []
        b_anns = []
        for ann in anns:
            if ann["time"] == -1:
                assert "_g_" in ann["question_id"]
                g_anns.append(ann)
            else:
                b_anns.append(ann)
        g_metrics = super().calculate_metrics(g_anns)
        b_metrics = super().calculate_metrics(b_anns)

        return {
            "global_yes_count": g_metrics["yes_count"],
            "global_no_count": g_metrics["no_count"],
            "global_accuracy": g_metrics["accuracy"],
            "global_average_score": g_metrics["average_score"],
            "breakpoint_yes_count": b_metrics["yes_count"],
            "breakpoint_no_count": b_metrics["no_count"],
            "breakpoint_accuracy": b_metrics["accuracy"],
            "breakpoint_average_score": b_metrics["average_score"],
        }
