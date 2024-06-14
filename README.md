# Video VLM Eval

## Commands

- `scripts/run_inference_zero_shot_qa.py`: run inference on a zero-shot qa dataset. supports distributed inference via `torchrun`.
- `scripts/run_eval_zero_shot_qa.py`: run evaluation on the inference results on a zero-shot qa dataset.

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

## Models

### Video-ChatGPT

```bash
<command> \
---model_name_or_path /path/to/video/chatgpt/output/dir \
... other arguments
```
