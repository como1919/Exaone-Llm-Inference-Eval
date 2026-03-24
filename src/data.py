import glob
import json
import os
import re
from typing import Dict, List, Tuple

import pandas as pd


def extract_inst_and_label(text: str) -> Tuple[str, str]:
    pattern_content = r"<s>\[INST\](.*?)\[/INST\]"
    pattern_label = r"\[/INST\](.*?)</s>"

    content_match = re.search(pattern_content, text, re.DOTALL)
    label_match = re.search(pattern_label, text, re.DOTALL)

    content = content_match.group(1).strip() if content_match else ""
    label = label_match.group(1).strip() if label_match else ""
    return content, label


def load_test_data_from_single_tsv(file_path: str):
    df = pd.read_csv(file_path, sep="\t", header=None, names=["text"])

    input_list = []
    label_list = []
    json_label_list = []

    for idx, row in df.iterrows():
        content, label = extract_inst_and_label(row["text"])
        input_list.append(content)
        label_list.append(label)

        try:
            parsed = json.loads(label)
        except json.JSONDecodeError:
            parsed = None

        json_label_list.append({
            "index": idx,
            "raw_label": label,
            "data": parsed,
        })

    return input_list, label_list, json_label_list


def load_test_data_from_directory(directory_path: str) -> Tuple[Dict[str, str], List[dict]]:
    all_files = glob.glob(os.path.join(directory_path, "*.tsv"))
    df_list = []

    for file in all_files:
        temp_df = pd.read_csv(file, sep="\t")
        if "file_name" not in temp_df.columns or "data" not in temp_df.columns:
            raise ValueError(f"{file} must contain 'file_name' and 'data' columns.")
        df_list.append(temp_df)

    df = pd.concat(df_list, ignore_index=True)

    input_dict = {}
    json_label_list = []

    for _, row in df.iterrows():
        file_name = row["file_name"]
        text = row["data"]

        content, label = extract_inst_and_label(text)
        input_dict[file_name] = content

        try:
            parsed = json.loads(label)
        except json.JSONDecodeError:
            parsed = None

        json_label_list.append({
            "index": file_name,
            "raw_label": label,
            "data": parsed,
        })

    return input_dict, json_label_list