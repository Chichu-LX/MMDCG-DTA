"""Command-line entry point for the paper's three-stage training curriculum."""

import argparse
from pathlib import Path

import yaml

from Data.training import run_pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("default.yaml"))
    parser.add_argument("--data-dir", type=Path, default=Path("Data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/training"))
    parser.add_argument("--stage", choices=("all", "1", "2", "3"), default="all")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    run_pipeline(config, args.data_dir, args.output_dir, args.stage)


if __name__ == "__main__":
    main()
