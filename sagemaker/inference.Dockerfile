FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.2.0-gpu-py310-cu121-ubuntu20.04-sagemaker

ENV PYTHONUNBUFFERED=1 \
    # Poetry
    # https://python-poetry.org/docs/configuration/#using-environment-variables
    # make poetry install to this location
    POETRY_HOME="/opt/poetry" \
    # do not ask any interactive question
    POETRY_NO_INTERACTION=1 \
    # never create virtual environment automatically, only use env prepared by us
    POETRY_VIRTUALENVS_CREATE=false

# install poetry - respects $POETRY_VERSION & $POETRY_HOME
# The --mount will mount the buildx cache directory to where
# Poetry and Pip store their cache so that they can re-use it
RUN --mount=type=cache,target=/root/.cache \
    curl -sSL https://install.python-poetry.org | python -

# prepend poetry to path
ENV PATH="$POETRY_HOME/bin:$PATH"

# Install dependencies using poetry
COPY poetry.lock pyproject.toml /opt/ml/code/
RUN --mount=type=cache,target=/root/.cache \
    --mount=type=secret,id=gh_priv_key,target=/root/.ssh/gh_priv_key \
    cd /opt/ml/code && GIT_SSH_COMMAND="ssh -i /root/.ssh/gh_priv_key" poetry install --no-root

# Manual Flash Attention 2 Installation
# Note: Prismatic doesn't support flash attention for inference, but let's install it for future.
RUN pip install packaging ninja
RUN pip install flash-attn --no-build-isolation

# Set Sagemaker Environment Variables =>> Define `run_inference.py` as entrypoint!
ENV SAGEMAKER_SUBMIT_DIRECTORY=/opt/ml/code
ENV SAGEMAKER_PROGRAM=/opt/ml/code/scripts/run_inference.py

# Copy code to `/opt/ml/code`
COPY README.md /opt/ml/code/
COPY video_vlm_eval /opt/ml/code/video_vlm_eval
COPY scripts /opt/ml/code/scripts
RUN cd /opt/ml/code && poetry install --only-root --extras=flash-attn
