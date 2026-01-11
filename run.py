import torch
import pathway as pw
import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from sklearn.metrics import f1_score
import re

# ===============================
# DEVICE
# ===============================
DEVICE = 0 if torch.cuda.is_available() else -1
DTYPE = torch.float16 if DEVICE != -1 else torch.float32

# ===============================
# MODELS
# ===============================
RETRIEVER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
NLI_MODELS = {
    "bart": "facebook/bart-large-mnli",
    "deberta": "microsoft/deberta-large-mnli"
}

retriever = SentenceTransformer(RETRIEVER_MODEL)

nli_pipes = {
    name: pipeline(
        "text-classification",
        model=model,
        device=DEVICE,
        torch_dtype=DTYPE
    )
    for name, model in NLI_MODELS.items()
}

# ===============================
# PARAMETERS
# ===============================
CHUNK_SIZE = 900
OVERLAP = 250
TOP_K = 5

# ===============================
# TEXT UTILS
# ===============================
def chunk_text(text):
    chunks, i = [], 0
    while i < len(text):
        chunks.append(text[i:i+CHUNK_SIZE])
        i += CHUNK_SIZE - OVERLAP
    return chunks

def extract_claims(text):
    parts = re.split(r"[.;]", text)
    return [p.strip() for p in parts if len(p.strip()) > 25]

# ===============================
# LOAD BOOKS
# ===============================
def load_books():
    books = {}
    for path in Path("books").glob("*.txt"):
        books[path.stem.replace("_", " ")] = path.read_text(
            encoding="utf-8", errors="ignore"
        )
    return books

books = load_books()

rows = []
for book, text in books.items():
    for c in chunk_text(text):
        rows.append({"book": book, "text": c})

pw_table = pw.debug.table_from_pandas(pd.DataFrame(rows))

chunk_embeddings = retriever.encode(
    [r["text"] for r in rows],
    convert_to_tensor=True
)

# ===============================
# GPU BATCHED NLI
# ===============================
def batched_nli(claims, evidences):
    """
    claims: list[str]
    evidences: list[str]
    returns dict[(i,j)] -> {model: (label, score)}
    """
    pairs = [
        f"{e} </s></s> {c}"
        for c in claims
        for e in evidences
    ]

    results = {}
    for name, pipe in nli_pipes.items():
        outputs = pipe(pairs, truncation=True, batch_size=16)
        idx = 0
        for i in range(len(claims)):
            for j in range(len(evidences)):
                results.setdefault((i, j), {})[name] = (
                    outputs[idx]["label"],
                    outputs[idx]["score"]
                )
                idx += 1
    return results

# ===============================
# CLAIM VALIDATION
# ===============================
def validate_claims(book, claims):
    claim_emb = retriever.encode(claims, convert_to_tensor=True)
    sims = util.cos_sim(claim_emb, chunk_embeddings).cpu().numpy()

    evidence_idxs = [
        np.argsort(sims[i])[-TOP_K:]
        for i in range(len(claims))
    ]

    evidences = list({
        rows[idx]["text"]
        for idxs in evidence_idxs
        for idx in idxs
        if rows[idx]["book"] == book
    })

    if not evidences:
        return [1] * len(claims), [], "No contradiction found"

    nli_results = batched_nli(claims, evidences)

    verdicts = []
    explanations = []

    for i, claim in enumerate(claims):
        contradicted = False
        best_entail = 0

        for j, ev in enumerate(evidences):
            res = nli_results.get((i, j), {})
            for label, score in res.values():
                if label == "CONTRADICTION" and score > 0.85:
                    verdicts.append(0)
                    explanations.append((claim, ev[:400], "Semantic contradiction"))
                    contradicted = True
                    break
                if label == "ENTAILMENT":
                    best_entail = max(best_entail, score)
            if contradicted:
                break

        if not contradicted:
            verdicts.append(1)
            explanations.append(
                (claim, evidences[0][:400], "No contradiction detected")
            )

    return verdicts, explanations, "Claim-level aggregation"

# ===============================
# AUTO THRESHOLD CALIBRATION
# ===============================
def calibrate(train_df):
    best_k, best_score = 1, 0

    for k in [1, 2, 3]:
        preds, gts = [], []
        for _, row in train_df.iterrows():
            claims = extract_claims(row["content"])
            verdicts, _, _ = validate_claims(row["book_name"], claims)
            pred = 0 if verdicts.count(0) >= k else 1
            gt = 1 if row["label"] == "consistent" else 0
            preds.append(pred)
            gts.append(gt)

        score = f1_score(gts, preds)
        if score > best_score:
            best_score = score
            best_k = k

    print(f"🔧 Calibrated contradiction threshold: {best_k}")
    return best_k

# ===============================
# MAIN
# ===============================
def main():
    Path("explanations").mkdir(exist_ok=True)

    CONTRADICTION_K = 2
    if Path("train.csv").exists():
        train = pd.read_csv("train.csv")
        CONTRADICTION_K = calibrate(train)

    test = pd.read_csv("test.csv")
    outputs = []

    for _, row in test.iterrows():
        claims = extract_claims(row["content"])
        verdicts, expl, _ = validate_claims(row["book_name"], claims)

        final = 0 if verdicts.count(0) >= CONTRADICTION_K else 1
        outputs.append(final)

        with open(f"explanations/example_{row['id']}.txt", "w") as f:
            for c, ev, reason in expl:
                f.write(f"CLAIM: {c}\n")
                f.write(f"EVIDENCE: {ev}\n")
                f.write(f"REASON: {reason}\n\n")
            f.write(f"FINAL VERDICT: {'CONSISTENT' if final else 'CONTRADICT'}")

    test["prediction"] = outputs
    test[["id", "prediction"]].to_csv("submission.csv", index=False)

    print("✅ submission.csv created")
    print("✅ Track B explanations generated")

if __name__ == "__main__":
    main()
