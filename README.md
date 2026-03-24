# EXAONE HF Inference & Evaluation Pipeline

This repository provides an end-to-end inference and evaluation pipeline for a fine-tuned and AWQ-quantized EXAONE 3.5 7.8B model deployed on Hugging Face.

---

## Overview

This project includes:
- Loading an AWQ-quantized model from Hugging Face
- Batch inference using vLLM
- Structured JSON output generation
- Merging batch outputs
- Evaluation using BLEU and ROUGE metrics

---

## Model

This project uses a fine-tuned EXAONE 3.5 7.8B model that was quantized using AWQ in a separate training pipeline.

- **Model**: EXAONE 3.5 7.8B (Instruct)
- **Quantization**: 4-bit AWQ
- **Architecture**: Custom ExaoneForCausalLM
- **Context Length**: up to 32K tokens

The model is optimized for efficient inference using vLLM.

---

## Pipeline

Raw Clinical Text (TSV)
        ↓
Prompt Parsing ([INST] format)
        ↓
vLLM Inference (AWQ model)
        ↓
Batch JSON Outputs
        ↓
Merge Outputs
        ↓
BLEU / ROUGE Evaluation

---

## Project Structure

exaone-hf-inference-eval/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── configs/
│   └── exaone_awq.yaml
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── model.py
│   ├── inference.py
│   ├── metrics.py
│   └── io_utils.py
├── scripts/
│   ├── run_inference.py
│   ├── merge_outputs.py
│   └── evaluate_outputs.py
└── outputs/
    └── .gitkeep

---

## Input Format

Each row should follow:

<s>[INST] input_text [/INST] output_json </s>

---

## Example

### Input
<s>[INST] F/27 DM/HTN(-/-) ... [/INST]

### Output
{
  "general_medical_history": "...",
  "recent_history": "..."
}

---

## Installation

pip install -r requirements.txt
cp .env.example .env

---

## Environment Variables

HF_TOKEN=your_huggingface_token_here

---

## ▶Run Inference

python scripts/run_inference.py \
  --config configs/exaone_awq.yaml \
  --input_path data/test_data \
  --output_dir outputs/exaone_awq \
  --input_mode directory \
  --batch_size 50

---

## Merge Outputs

python scripts/merge_outputs.py \
  --input_dir outputs/exaone_awq \
  --output_file outputs/exaone_awq_merged.json \
  --output_key results

---

## Evaluate Outputs

python scripts/evaluate_outputs.py \
  --label_input_path data/test_data \
  --generated_json outputs/exaone_awq_merged.json \
  --label_input_mode directory

---

## Evaluation Metrics

- BLEU
- ROUGE

---

## Output Format

{
  "index": "sample_001.txt",
  "data": {
    "general_medical_history": "...",
    "recent_history": "..."
  },
  "error_sample": null
}

---

## Notes

- Training data is not included.
- This repo focuses on inference and evaluation.
- Clinical data should be handled securely.

---
