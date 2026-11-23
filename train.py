import argparse
import os
from pathlib import Path
import subprocess
import sys
from gtha_labor_marl.curriculum import load_profile


def main():
    parser = argparse.ArgumentParser(
        description="Train the worker and firm policies through three curriculum stages."
    )
    parser.add_argument(
        "--data", type=Path, required=True, help="Input directory containing Data and Data_Agents"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Output directory for one policy seed"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--demo", action="store_true", help="Use a small population and one short update per stage"
    )
    args = parser.parse_args()
    data = args.data.resolve()
    if not (data / "Data").is_dir() or not (data / "Data_Agents").is_dir():
        parser.error(f"Missing input directories under {data}. Generate the synthetic data first.")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error(
            f"Choose an empty output directory to keep each training run separate: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["METRO_DATA_ROOT"] = str(data)
    if args.device == "cpu":
        environment.setdefault("OMP_NUM_THREADS", "2")
        environment.setdefault("MKL_NUM_THREADS", "2")
    module = Path(__file__).resolve().parent / "gtha_labor_marl" / "train_stage.py"
    for stage in (1, 2, 3):
        profile = load_profile(stage)
        steps = 8 if args.demo else profile["rollout_steps"]
        updates = 1 if args.demo else profile["updates"]
        destination = output / f"stage{stage}"
        command = [
            sys.executable,
            str(module),
            "--seed",
            str(args.seed),
            "--stage",
            str(stage),
            "--n_workers",
            "250" if args.demo else "1000",
            "--n_firms",
            "80",
            "--total_timesteps",
            str(steps * updates),
            "--rollout_steps",
            str(steps),
            "--device",
            args.device,
            "--output_dir",
            str(destination),
        ]
        if stage > 1:
            command.extend(
                ["--resume_from", str(output / f"stage{stage - 1}" / "final_checkpoint.pt")]
            )
        destination.mkdir()
        subprocess.run(command, env=environment, cwd=destination, check=True)
        checkpoint = destination / "final_checkpoint.pt"
        if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
            raise RuntimeError(f"Training ended without producing {checkpoint}")


if __name__ == "__main__":
    main()
