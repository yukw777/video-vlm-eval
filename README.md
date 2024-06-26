# Video VLM Eval

## Commands

- `scripts/run_inference.py`: run inference on various benchmarks. supports distributed inference via `torchrun`.
- `scripts/run_eval.py`: run various evaluations.
- `scripts/run_eval_video_chatgpt.py`: run Video-ChatGPT style evaluation.

## Datasets

### ActivityNet-QA

```bash
<command> \
--dataset video_vlm_eval.ActivityNetQADataset \
--dataset.video_dirs '[/path/to/v1-2/test, /path/to/v1-3/test]' \
--dataset.gt_file_question /path/to/test_q.json \
--dataset.gt_file_answer /path/to/test_a.json \
... other arguments
```

### MSVD-QA

```bash
<command> \
--dataset video_vlm_eval.MSVDQADataset \
--dataset.video_dir /path/to/YouTubeClips \
--dataset.video_mapping_file /path/to/youtube_mapping.txt \
--dataset.annotations_file /path/to/test_qa.json \
... other arguments
```

### MSRVTT-QA

```bash
<command> \
--dataset video_vlm_eval.MSRVTTQADataset \
--dataset.video_dir /path/to/videos/all \
--dataset.annotations_file /path/to/test_qa.json \
... other arguments
```

### TGIF-QA Frame

```bash
<command> \
--dataset video_vlm_eval.TGIFQAFrameDataset \
--dataset.video_dir /path/to/converted/videos \
--dataset.frame_annotations_file /path/to/frameqa_question.csv \
... other arguments
```

### Video-ChatGPT Generic/Temporal

```bash
<command> \
--dataset video_vlm_eval.VideoChatGPTGeneralDataset \
--dataset.video_dir /path/to/Test_Videos/ \
--dataset.annotations_file /path/to/generic/or/temporal_qa.json \
... other arguments
```

### Video-ChatGPT Consistency

```bash
<command> \
--dataset video_vlm_eval.VideoChatGPTConsistencyDataset \
--dataset.video_dir /path/to/Test_Videos/ \
--dataset.annotations_file /path/to/consistency_qa.json \
... other arguments
```

### EgoSchema

```bash
<command> \
--dataset video_vlm_eval.EgoSchemaDataset \
--dataset.video_dir /path/to/videos \
--dataset.question_file /path/to/questions.json \
--dataset.answer_file /path/to/subset_answers.json \
... other arguments
```

## Tasks

### Video-ChatGPT Zero-Shot QA Task

```bash
<command> \
--task video_vlm_eval.VideoChatGPTZeroShotQATask \
... other arguments
```

### Video-ChatGPT Correctness Task

```bash
<command> \
--task video_vlm_eval.VideoChatGPTCorrectnessTask \
... other arguments
```

### Video-ChatGPT Detailed Orientation Task

```bash
<command> \
--task video_vlm_eval.VideoChatGPTDetailedOrientationTask \
... other arguments
```

### Video-ChatGPT Context Task

```bash
<command> \
--task video_vlm_eval.VideoChatGPTContextTask \
... other arguments
```

### Video-ChatGPT Temporal Task

```bash
<command> \
--task video_vlm_eval.VideoChatGPTTemporalTask \
... other arguments
```

### Video-ChatGPT Consistency Task

```bash
<command> \
--task video_vlm_eval.VideoChatGPTConsistencyTask \
... other arguments
```

### Multiple Choice Task

```bash
<command> \
--task video_vlm_eval.MultipleChoice \
... other arguments
```

### EgoSchema Multiple Choice Task

```bash
<command> \
--task video_vlm_eval.EgoSchemaMultipleChoice \
... other arguments
```

## Models

### Prismatic Model for Zero-Shot QA

```bash
<command> \
--model video_vlm_eval.PrismaticZeroShotQAModel \
--model.model_name_or_path /path/to/model/output/dir \
--model.dtype {float16, bfloat16} \
... other arguments
```

### Prismatic Model for Video-ChatGPT Consistency

```bash
<command> \
--model video_vlm_eval.PrismaticVideoChatGPTConsistencyModel \
--model.model_name_or_path /path/to/model/output/dir \
--model.dtype {float16, bfloat16} \
... other arguments
```

### Prismatic Model for EgoSchema

```bash
<command> \
--model video_vlm_eval.PrismaticEgoSchemaModel \
--model.model_name_or_path /path/to/model/output/dir \
--model.dtype {float16, bfloat16} \
... other arguments
```
