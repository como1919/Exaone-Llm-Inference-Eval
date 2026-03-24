import argparse
import yaml
from src.io_utils import merge_batch_outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    merge_batch_outputs(
        input_dir=args.input_dir,
        output_file_path=args.output_file,
        output_key=cfg["output"]["output_key"],
    )


if __name__ == "__main__":
    main()