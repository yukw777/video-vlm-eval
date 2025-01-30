import torch
from jsonargparse import CLI

from video_vlm_eval import Model
from torch.utils.data import default_collate


def main(
    model: Model,
    video_path: str,
    prompt: str = "Can you summarize the video?",
    device: str = "cuda",
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

    # Perform inference
    with torch.no_grad(), torch.autograd.profiler.profile(use_cuda=True) as prof:
        model.perform(preprocessed, max_new_tokens=1)
    print(prof)

    # Calculate peak memory usage during inference
    peak_memory = torch.cuda.max_memory_allocated()
    print(f"Peak memory usage: {peak_memory / (1024 ** 2):.2f} MB")


if __name__ == "__main__":
    CLI(main, as_positional=False)
