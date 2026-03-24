from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge


def calculate_metrics(
    label_data,
    generated_data,
    field_names=("general_medical_history", "recent_history"),
):
    rouge = Rouge()
    metrics = {}

    valid_samples = 0

    label_map = {item["index"]: item for item in label_data}

    for field in field_names:
        references = []
        hypotheses = []

        for gen_entry in generated_data:
            if gen_entry.get("error_sample") is not None:
                continue

            sample_id = gen_entry["index"]
            label_entry = label_map.get(sample_id)

            if not label_entry:
                continue
            if label_entry["data"] is None:
                continue

            gen_text = gen_entry["data"].get(field, "")
            label_text = label_entry["data"].get(field, "")

            if gen_text and label_text:
                references.append([label_text.split()])
                hypotheses.append(gen_text.split())

        if hypotheses:
            bleu_score = corpus_bleu(references, hypotheses)
            rouge_scores = rouge.get_scores(
                [" ".join(h) for h in hypotheses],
                [" ".join(r[0]) for r in references],
                avg=True,
            )
        else:
            bleu_score = 0.0
            rouge_scores = {}

        metrics[f"BLEU_{field}"] = bleu_score
        metrics[f"ROUGE_{field}"] = rouge_scores

    for gen_entry in generated_data:
        if gen_entry.get("error_sample") is None:
            valid_samples += 1

    metrics["valid_samples"] = valid_samples
    return metrics