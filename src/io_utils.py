import glob
import json
import os
import re


def merge_batch_outputs(input_dir: str, output_file_path: str, output_key: str = "results"):
    batch_files = glob.glob(os.path.join(input_dir, "batch_*.json"))
    batch_files = sorted(
        batch_files,
        key=lambda x: int(re.search(r"batch_(\d+)\.json$", os.path.basename(x)).group(1))
    )

    merged = []
    for file_path in batch_files:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        merged.extend(data.get("results", []))

    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump({output_key: merged}, f, ensure_ascii=False, indent=2)

    print(f"Merged {len(batch_files)} files into {output_file_path}")