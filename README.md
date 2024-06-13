# Video VLM Eval

## Inference

Run `scripts/run_inference_zero_shot_qa.py` with appropriate arguments. It supports distributed inference via `torchrun`.

### Datasets

#### ActivityNet-QA

```bash
{python,torchrun} scripts/run_inference_zero_shot_qa.py \
--dataset video_vlm_eval.ActivityNetQADataset \
--dataset.video_dirs '[/path/to/v1-2/test, /path/to/v1-3/test]' \
--dataset.gt_file_question /path/to/test_q.json \
... other arguments
```

#### MSVD-QA

```bash
{python,torchrun} scripts/run_inference_zero_shot_qa.py \
--dataset video_vlm_eval.MSVDQADataset \
--dataset.video_dir /path/to/YouTubeClips \
--dataset.video_mapping_file /path/to/youtube_mapping.txt \
--dataset.annotations_file /path/to/test_qa.json \
... other arguments
```

### Models

#### Video-ChatGPT

```bash
{python,torchrun} scripts/run_inference_zero_shot_qa.py \
---model_name_or_path /path/to/video/chatgpt/output/dir \
... other arguments
```
