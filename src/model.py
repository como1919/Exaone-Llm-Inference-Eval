import os
from dotenv import load_dotenv
import vllm


def load_vllm_model(
    model_id: str,
    quantization: str = "AWQ",
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 4000,
):
    load_dotenv()

    token = os.getenv("HF_TOKEN")
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token

    llm = vllm.LLM(
        model=model_id,
        quantization=quantization,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
    )

    return llm