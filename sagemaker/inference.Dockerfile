FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.2.0-gpu-py310-cu121-ubuntu20.04-sagemaker

# Install base dependencies
RUN pip uninstall -y transformer-engine
COPY base-requirements.txt /opt/ml/code/
RUN cd /opt/ml/code && pip install -r base-requirements.txt

# Manual Flash Attention 2 Installation
# Note: Prismatic doesn't support flash attention for inference, but let's install it for future.
RUN pip install packaging ninja
RUN pip install flash-attn==2.5.9.post1 --no-build-isolation

# Manual Prismatic Installation
RUN --mount=type=secret,id=gh_priv_key,target=/root/.ssh/gh_priv_key \
    GIT_SSH_COMMAND="ssh -i /root/.ssh/gh_priv_key" pip install git+ssh://git@github.com/yukw777/prismatic-dev.git@285e8203a2dacf600012aa58b36e658e2c466bb3

# Set Sagemaker Environment Variables =>> Define `run_inference.py` as entrypoint!
ENV SAGEMAKER_SUBMIT_DIRECTORY=/opt/ml/code
ENV SAGEMAKER_PROGRAM=/opt/ml/code/scripts/run_inference.py

# Copy code to `/opt/ml/code`
COPY pyproject.toml README.md /opt/ml/code/
COPY video_vlm_eval /opt/ml/code/video_vlm_eval
COPY scripts /opt/ml/code/scripts
RUN cd /opt/ml/code && pip install -e .
