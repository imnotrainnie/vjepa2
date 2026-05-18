# Implements EK_PLAN_PART4A §12.2
import argparse
import os
from multiprocessing import Process
from pathlib import Path
from typing import List

import yaml

from src.utils.distributed import init_distributed


def str2bool(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def parse_args():
    parser = argparse.ArgumentParser(description="Launcher for EK multimodal action prediction")
    parser.add_argument("--fname", type=Path, default=Path(__file__).parent / "ek_action_prediction.yaml")
    parser.add_argument("--devices", nargs="+", default=["cuda:0"])
    parser.add_argument("--debugmode", type=str2bool, default=False)
    parser.add_argument("--val_only", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    return parser.parse_args()


def load_params(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def process_main(rank: int, world_size: int, devices: List[str], params):
    device = devices[rank]
    if device.startswith("cuda"):
        os.environ["CUDA_VISIBLE_DEVICES"] = device.split(":")[-1]
    init_distributed(rank_and_world_size=(rank, world_size))
    from multimodal_evals.ek_action_prediction.eval import main as ek_main

    ek_main(args_eval=params)


def main():
    args = parse_args()
    params = load_params(args.fname)
    if args.val_only:
        params["val_only"] = True
    if args.checkpoint:
        params["resume"] = True
        params["resume_checkpoint"] = args.checkpoint
    if args.batch_size is not None:
        params.setdefault("data", {})["batch_size"] = args.batch_size

    devices = args.devices
    if args.debugmode or len(devices) == 1:
        process_main(rank=0, world_size=1, devices=devices, params=params)
        return

    procs = []
    for rank in range(len(devices)):
        proc = Process(target=process_main, args=(rank, len(devices), devices, params))
        proc.start()
        procs.append(proc)
    for proc in procs:
        proc.join()
        if proc.exitcode != 0:
            raise RuntimeError(f"Rank process exited with code {proc.exitcode}")


if __name__ == "__main__":
    main()
