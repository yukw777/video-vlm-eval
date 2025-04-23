import torch
from jsonargparse import CLI

from video_vlm_eval import Model
from torch.utils.data import default_collate
from tqdm import tqdm
import numpy as np


def main(
    model: Model,
    video_path: str,
    prompt: str = "Can you summarize the video?",
    device: str = "cuda",
    repeat: int = 10,
    warmup: int = 2,
) -> None:
    model.to(device)
    preprocessed = model.preprocess({"video_path": video_path, "question": prompt})
    if model.collate_fn is None:
        preprocessed = default_collate([preprocessed])
    else:
        preprocessed = model.collate_fn([preprocessed])

    for k, v in preprocessed.items():
        if hasattr(v, "to"):
            preprocessed[k] = v.to(device)

    # Synchronize GPU and CPU for accurate timing
    torch.cuda.synchronize()

    # Clear any existing cache to get an accurate reading
    torch.cuda.empty_cache()

    for _ in tqdm(range(warmup), desc="Warm Up"):
        with torch.no_grad():
            model.perform(preprocessed, max_new_tokens=1)

    # Perform inference
    cpu_times = []
    cuda_times = []
    for i in tqdm(range(repeat)):
        print(f"===========Run {i}=============")
        with torch.no_grad(), torch.autograd.profiler.profile(use_cuda=True) as prof:
            model.perform(preprocessed, max_new_tokens=1)
        cpu_times.append(prof.self_cpu_time_total)
        print(f"Self CPU time total: {prof.self_cpu_time_total}")
        total_cuda_time = sum(evt.self_cuda_time_total for evt in prof.function_events)
        cuda_times.append(total_cuda_time)
        print(f"Self CUDA time total: {total_cuda_time}")
        print("===============================")

    avg_cpu_time = float(np.array(cpu_times).mean())
    print(f"Average self CPU time total: {avg_cpu_time / 1000:.2f}ms")
    avg_cuda_time = float(np.array(cuda_times).mean())
    print(f"Average self CUDA time total: {avg_cuda_time / 1000:.2f}ms")
    # Calculate peak memory usage during inference
    peak_memory = torch.cuda.max_memory_allocated()
    print(f"Peak memory usage: {peak_memory / (1024**2):.2f} MB")


if __name__ == "__main__":
    CLI(main, as_positional=False)
