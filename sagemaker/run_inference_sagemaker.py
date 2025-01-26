import boto3
from sagemaker.pytorch import PyTorch
import sagemaker
from sagemaker.batch_queueing.queue import Queue


class BotoSession(boto3.Session):
    def __init__(
        self,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_session_token: str | None = None,
        region_name: str | None = None,
        profile_name: str | None = None,
    ) -> None:
        super().__init__(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            region_name=region_name,
            profile_name=profile_name,
        )


def run(
    job_name: str,
    wandb_name: str,
    wandb_api_key: str,
    output_path: str,
    s3_data_paths: list[str],
    run_inference_args: list[tuple[str, str]],
    role_arn: str = "arn:aws:iam::124224456861:role/service-role/SageMaker-SageMakerAllAccess",
    instance_count: int = 1,
    instance_type: str = "ml.g5.12xlarge",
    image_uri: str = "124224456861.dkr.ecr.us-east-1.amazonaws.com/peter.yu-video-vlm-eval-inference:latest",
    max_hours: int = 24,
    tags: dict[str, str] | None = None,
    boto_session: BotoSession | None = None,
    use_reserved_capacity: bool = False,
    s3_hf_home: str | None = None,
    use_queue: bool = True,
    queue_priority: int = 10,
    queue_fss_identifier: str = "default",
) -> None:
    env_vars = {
        "WANDB_API_KEY": wandb_api_key,
        "WANDB_NAME": wandb_name,
        "SM_USE_RESERVED_CAPACITY": "1" if use_reserved_capacity else "0",
    }
    if s3_hf_home is not None:
        s3_data_paths.append(s3_hf_home)
        env_vars["HF_HOME"] = f"/opt/ml/input/data/data_{len(s3_data_paths) -1}/"
    sagemaker_session = sagemaker.Session(boto_session=boto_session)  # type: ignore
    sagemaker_session.boto_region_name
    estimator = PyTorch(
        "scripts/run_inference.py",
        role=role_arn,
        base_job_name=job_name,
        instance_count=instance_count,
        instance_type=instance_type,
        image_uri=image_uri,
        hyperparameters=dict(run_inference_args),
        environment=env_vars,
        sagemaker_session=sagemaker_session,
        keep_alive_period_in_seconds=3600,
        max_run=60 * 60 * max_hours,
        distribution={"torch_distributed": {"enabled": True}},
        disable_profiler=True,
        output_path=output_path,
        # set code_location to output_path so as not to clutter the s3 bucket.
        # Note that code_location shouldn't have a trailing slash
        code_location=output_path
        if not output_path.endswith("/")
        else output_path[:-1],
        input_mode="FastFile",
        tags=[{"Key": k, "Value": v} for k, v in tags.items()]
        if tags is not None
        else None,
    )
    inputs = {
        # input paths must have a trailing slash
        f"data_{i}": path if path.endswith("/") else path + "/"
        for i, path in enumerate(s3_data_paths)
    }
    if use_queue:
        queue = Queue(
            f'fss-{instance_type.replace(".", "-")}-{sagemaker_session.boto_region_name}'
        )
        print(f"Starting training job on queue {queue.queue_name}")
        queued_job = queue.submit(
            estimator,
            inputs,
            job_name=job_name,
            priority=queue_priority,
            share_identifier=queue_fss_identifier,
            timeout={"attemptDurationSeconds": 60 * 60 * max_hours},
        )
        print(f"Queued job: {queued_job}")
    else:
        estimator.fit(inputs=inputs)


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(run, as_positional=False)
