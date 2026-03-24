import json
import os
from typing import Dict, List, Union

from tqdm import tqdm
from vllm import SamplingParams


def run_model_in_batches_and_save(
    llm,
    inputs: Union[List[str], Dict[str, str]],
    output_dir: str,
    batch_size: int = 50,
    max_tokens: int = 4000,
    temperature: float = 0.2,
    top_p: float = 0.8,
    stop: list | None = None,
):
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = llm.get_tokenizer()
    bos_token = tokenizer.bos_token or ""
    eos_token = tokenizer.eos_token or ""

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=stop or ["\n[|assistant|]"],
    )

    if isinstance(inputs, dict):
        items = list(inputs.items())
    else:
        items = list(enumerate(inputs))

    total = len(items)
    total_batches = (total + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, total)
        batch_items = items[start:end]

        prompts = [f"{bos_token}{content}{eos_token}" for _, content in batch_items]
        outputs = llm.generate(prompts, sampling_params)

        batch_results = []
        for (sample_id, _), output in zip(batch_items, outputs):
            text = output.outputs[0].text.replace("[|assistant|]", "").strip()

            entry = {
                "index": sample_id,
                "data": {
                    "general_medical_history": None,
                    "recent_history": None,
                },
                "error_sample": None,
            }

            try:
                parsed = json.loads(text)
                entry["data"]["general_medical_history"] = parsed.get("general_medical_history", "")
                entry["data"]["recent_history"] = parsed.get("recent_history", "")
            except json.JSONDecodeError:
                entry["error_sample"] = text

            batch_results.append(entry)

        batch_path = os.path.join(output_dir, f"batch_{batch_idx + 1}.json")
        with open(batch_path, "w", encoding="utf-8") as f:
            json.dump({"results": batch_results}, f, ensure_ascii=False, indent=2)

        print(f"Saved {batch_path}")