import argparse
import json
import yaml

from src.data import load_test_data_from_directory, load_test_data_from_single_tsv
from src.metrics import calculate_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label_input_path", required=True)
    parser.add_argument("--generated_json", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--label_input_mode", choices=["single_tsv", "directory"], default="single_tsv")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # === label ===
    if args.label_input_mode == "single_tsv":
        _, _, label_data = load_test_data_from_single_tsv(args.label_input_path)
    else:
        _, label_data = load_test_data_from_directory(
            args.label_input_path,
            text_column=cfg["data"]["text_column"],
            file_name_column=cfg["data"]["file_name_column"],
        )

    # === prediction ===
    with open(args.generated_json, "r", encoding="utf-8") as f:
        generated = json.load(f)

    generated_data = generated.get(cfg["output"]["output_key"])
    if generated_data is None:
        raise KeyError(f"Missing key: {cfg['output']['output_key']}")

    metrics = calculate_metrics(label_data, generated_data)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()