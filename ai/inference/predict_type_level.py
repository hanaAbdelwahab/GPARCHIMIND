import json
import os
import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from ai.utils.nfr_mapping import NFR_MAP_REVERSE
from infrastructure.repositories.nfr_dataset_repository import (
    NFRDatasetRepository,
    NFRPredictionRepository
)

BASE_OUTPUT = "data/outputs"
NFR_INPUT_PATH = os.path.join(BASE_OUTPUT, "non_functional_requirements.json")
OUTPUT_PATH = os.path.join(BASE_OUTPUT, "nfr_predictions_type_level.json")

MODEL_TYPE_PATH = "hanawahab/trained_nfr_type_model"
MODEL_LEVEL_PATH = "hanawahab/trained_nfr_level_model"

# ============================================================
# LOAD EVERYTHING ONCE AT STARTUP
# ============================================================

print("🔥 Loading dataset for encoders...")
_df_global = NFRDatasetRepository.load_nfr_dataset_from_mongo()

le_type = LabelEncoder()
le_type.fit(_df_global["Type"])

le_level = LabelEncoder()
le_level.fit(_df_global["Level"])

print("✅ Encoders ready")

print("🔥 Loading TYPE model...")
tokenizer_type = BertTokenizer.from_pretrained(MODEL_TYPE_PATH)
model_type = BertForSequenceClassification.from_pretrained(MODEL_TYPE_PATH)
model_type.eval()
print("✅ TYPE model loaded")

print("🔥 Loading LEVEL model...")
tokenizer_level = BertTokenizer.from_pretrained(MODEL_LEVEL_PATH)
model_level = BertForSequenceClassification.from_pretrained(MODEL_LEVEL_PATH)
model_level.eval()
print("✅ LEVEL model loaded")


# ============================================================
# HELPERS
# ============================================================

def softmax_np(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / (np.sum(e) + 1e-12)


# ============================================================
# PREDICT TYPE + LEVEL FOR ALL NFRs IN A PROJECT
# ============================================================

def predict_and_save_nfr(project_id: str):
    # Load extracted NFRs from file
    with open(NFR_INPUT_PATH, "r", encoding="utf-8") as f:
        nfrs = json.load(f)

    texts = [item["description"] for item in nfrs]
    if not texts:
        raise ValueError("No NFRs found")

    # Tokenize using module-level tokenizer
    tokens = tokenizer_type(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    # Predict using module-level model
    with torch.no_grad():
        logits_type = model_type(**tokens).logits

    logits_type_np = logits_type.cpu().numpy()

    results = []
    for i, item in enumerate(nfrs):
        probs = softmax_np(logits_type_np[i])
        pred_idx = int(np.argmax(probs))

        pred_type = le_type.inverse_transform([pred_idx])[0]
        confidence = float(probs[pred_idx])

        results.append({
            "title": item.get("title"),
            "description": item.get("description"),
            "predicted_type": pred_type,
            "predicted_type_label": NFR_MAP_REVERSE.get(pred_type, pred_type),
            "confidence": confidence,
            "predicted_level": "System",
            "confirmed": False
        })

    NFRPredictionRepository.save_batch(project_id, results)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✅ NFR predictions generated: {len(results)} total")
    return results


# ============================================================
# PREDICT LEVEL FOR A SINGLE TEXT (after user confirms type)
# ============================================================

def predict_level_for_text(text: str) -> str:
    tokens = tokenizer_level(
        [text],
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    with torch.no_grad():
        logits = model_level(**tokens).logits

    pred_id = int(torch.argmax(logits, dim=1))
    return le_level.inverse_transform([pred_id])[0]