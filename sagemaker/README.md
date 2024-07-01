# Running video_vlm_eval Scripts on SageMaker

## Building a Docker image

```bash
# Note that we need a private ssh key to install the internal version of prismatic.
# The key will not be saved with the image as we use the Docker secret functionality.
sudo docker build --secret id=gh_priv_key,src=/path/to/your/private/key -f sagemaker/inference.Dockerfile -t 124224456861.dkr.ecr.us-east-1.amazonaws.com/peter.yu-video-vlm-eval-inference .
```
