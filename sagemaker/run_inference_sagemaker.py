from sagemaker.inputs import FileSystemInput
from sagemaker.pytorch import PyTorch
import sagemaker
from typing import Any


def run(
    job_name: str,
    input_source: str,
    wandb_name: str,
    wandb_api_key: str,
    output_path: str,
    run_inference_args: list[tuple[str, str]],
    lustre_file_system_id: str = "",
    lustre_directory_path: str = "",
    s3_input_path: str = "",
    role_arn: str = "arn:aws:iam::124224456861:role/service-role/SageMaker-SageMakerAllAccess",
    instance_count: int = 1,
    instance_type: str = "ml.g5.12xlarge",
    image_uri: str = "124224456861.dkr.ecr.us-east-1.amazonaws.com/peter.yu-video-vlm-eval-inference:latest",
    subnets: tuple[str, ...] = (
        "subnet-07bf42d7c9cb929e4",
        "subnet-05f1115c7d6ccbd07",
        "subnet-0e260ba29726b9fbb",
    ),
    security_group_ids: tuple[str, ...] = (
        "sg-0afb9fb0e79a54061",
        "sg-0333993fea1aeb948",
        "sg-0c4b828f4023a04cc",
    ),
    max_hours: int = 24,
    tags: dict[str, str] | None = None,
) -> None:
    estimator_kwargs: dict[str, Any] = {}
    if tags is not None:
        estimator_kwargs["tags"] = [{"Key": k, "Value": v} for k, v in tags.items()]
    if input_source == "lustre":
        eval_fs = FileSystemInput(
            file_system_type="FSxLustre",
            file_system_access_mode="rw",
            file_system_id=lustre_file_system_id,
            directory_path=lustre_directory_path,
        )
    elif input_source == "s3":
        eval_fs = s3_input_path
        estimator_kwargs["input_mode"] = "FastFile"
    else:
        raise ValueError(
            f'Only "lustre" and "s3" are supported for input_source. Got "{input_source}"'
        )
    estimator = PyTorch(
        "scripts/run_inference.py",
        role=role_arn,
        base_job_name=job_name,
        instance_count=instance_count,
        instance_type=instance_type,
        image_uri=image_uri,
        hyperparameters=dict(run_inference_args),
        environment={"WANDB_API_KEY": wandb_api_key, "WANDB_NAME": wandb_name},
        sagemaker_session=sagemaker.Session(),  # type: ignore
        subnets=subnets,
        security_group_ids=security_group_ids,
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
        **estimator_kwargs,
    )
    estimator.fit(inputs={"training": eval_fs})


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(run, as_positional=False)
