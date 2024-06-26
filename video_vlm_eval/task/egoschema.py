from video_vlm_eval.task import MultipleChoice


class EgoSchemaMultipleChoice(MultipleChoice):
    def calculate_metrics(self, anns: list[dict]) -> dict:
        # calculate accuracy only for annotations with answers
        return super().calculate_metrics([ann for ann in anns if ann["answer"] != ""])
