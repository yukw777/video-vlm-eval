import pprint
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm


@dataclass
class Converter:
    out_dir_path: Path

    def convert(self, gif_path: Path) -> None:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(gif_path),
                "-vf",
                # this filter is to ensure the dimensions are even numbers,
                # which is a requirement of h.264.
                "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                "-c:v",
                "libx264",
                # spend more time to compress more!
                "-preset",
                "slow",
                f"{self.out_dir_path / gif_path.stem}.mp4",
            ],
            check=True,
            capture_output=True,
        )


def main(gif_dir: str, out_dir: str) -> None:
    gif_dir_path = Path(gif_dir)
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    # figure out which gif hasn't been converted yet
    converted_gifs = {converted.stem for converted in out_dir_path.glob("*.mp4")}
    gif_paths = [
        gif_path
        for gif_path in gif_dir_path.glob("*.gif")
        if gif_path.stem not in converted_gifs
    ]
    print(f"Converting {len(gif_paths)} gifs.")

    # convert
    converter = Converter(out_dir_path)
    with ThreadPoolExecutor() as executor:
        future_to_gif_path = {
            executor.submit(converter.convert, gif_path): gif_path
            for gif_path in gif_paths
        }
        not_finished: dict[str, str] = {}
        for future in tqdm(as_completed(future_to_gif_path), total=len(gif_paths)):
            try:
                future.result()
            except Exception as e:
                not_finished[future_to_gif_path[future].name] = str(e)
    if len(not_finished) != 0:
        pprint.pp(not_finished)
        exit(-1)
    exit()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gif_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    main(args.gif_dir, args.out_dir)
