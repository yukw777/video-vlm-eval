from video_vlm_eval.task import MultipleChoice


class VideoMMEMultiplechoice(MultipleChoice):
    def calculate_metrics(self, anns: list[dict]) -> dict:
        # calculate accuracy for overall, short, medium and long.
        short_anns = [ann for ann in anns if ann["duration"] == "short"]
        med_anns = [ann for ann in anns if ann["duration"] == "medium"]
        long_anns = [ann for ann in anns if ann["duration"] == "long"]
        assert len(anns) == len(short_anns) + len(med_anns) + len(long_anns)

        overall_acc = super().calculate_metrics(anns)["accuracy"]
        short_acc = super().calculate_metrics(short_anns)["accuracy"]
        med_acc = super().calculate_metrics(med_anns)["accuracy"]
        long_acc = super().calculate_metrics(long_anns)["accuracy"]

        return {
            "overall_accuracy": overall_acc,
            "short_accuracy": short_acc,
            "medium_accuracy": med_acc,
            "long_accuracy": long_acc,
        }
