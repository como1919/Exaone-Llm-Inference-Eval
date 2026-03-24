import argparse
import yaml

from src.data import (
    load_test_data_from_directory,
    load_test_data_from_single_tsv,
)
from src.inference import run_model_in_batches_and_save
from src.model import load_vllm_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--input_mode", choices=["single_tsv", "directory"], default="single_tsv")
    parser.add_argument("--batch_size", type=int, default=50)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # === model ===
    llm = load_vllm_model(
        model_id=cfg["model"]["model_id"],
        quantization=cfg["model"]["quantization"],
        gpu_memory_utilization=cfg["model"]["gpu_memory_utilization"],
        max_model_len=cfg["model"]["max_model_len"],
    )

    # === data ===
    if args.input_mode == "single_tsv":
        inputs, _, _ = load_test_data_from_single_tsv(args.input_path)
    else:
        inputs, _ = load_test_data_from_directory(
            args.input_path,
            text_column=cfg["data"]["text_column"],
            file_name_column=cfg["data"]["file_name_column"],
        )

    # === inference ===
    run_model_in_batches_and_save(
        llm=llm,
        inputs=inputs,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_tokens=cfg["generation"]["max_tokens"],
        temperature=cfg["generation"]["temperature"],
        top_p=cfg["generation"]["top_p"],
        stop=cfg["generation"]["stop"],
    )


if __name__ == "__main__":
    main()