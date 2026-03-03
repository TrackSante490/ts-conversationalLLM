import os
from typing import List, Optional
from datasets import load_dataset
from langchain_core.documents import Document

HF_DATASET = os.getenv("HF_DATASET", "Medilora/us_medical_license_exam_textbooks_en")
HF_SPLIT = os.getenv("HF_SPLIT", "train")
HF_MAX_ROWS = int(os.getenv("HF_MAX_ROWS", "500")) #testing with 500
HF_TEXT_FIELD = os.getenv("HF_TEXT_FIELD", "").strip() 


def _pick_text_field(sample: dict) -> str:
    """Pick the best text field from a dataset row (longest string)."""
    if HF_TEXT_FIELD and HF_TEXT_FIELD in sample:
        return HF_TEXT_FIELD

    candidates = []
    for k, v in sample.items():
        if isinstance(v, str):
            candidates.append((k, len(v.strip())))
    if not candidates:
        raise RuntimeError(f"No string fields found. Keys: {list(sample.keys())}")

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]


def load_hf_kb(
    dataset_name: Optional[str] = None,
    split: Optional[str] = None,
    max_rows: Optional[int] = None,
    text_field: Optional[str] = None,
) -> List[Document]:
    ds_name = dataset_name or HF_DATASET
    ds_split = split or HF_SPLIT
    limit = max_rows if max_rows is not None else HF_MAX_ROWS

    print(f"HF_DATASET: {ds_name}")
    print(f"HF_SPLIT: {ds_split}")
    print(f"HF_MAX_ROWS: {limit}")

    ds = load_dataset(ds_name, split=ds_split)
    if len(ds) == 0:
        raise RuntimeError("Dataset split is empty.")

    sample = ds[0]
    field = text_field or _pick_text_field(sample)

    print("Columns:", ds.column_names)
    print("Using text field:", field)

    documents: List[Document] = []
    total = min(limit, len(ds))
    empty_count = 0

    for i in range(total):
        txt = ds[i].get(field)
        if not isinstance(txt, str):
            continue
        txt = txt.strip()
        if not txt:
            empty_count += 1
            continue

        documents.append(
            Document(
                page_content=txt,
                metadata={
                    "source": "huggingface",
                    "dataset": ds_name,
                    "split": ds_split,
                    "row": i,
                    "field": field,
                },
            )
        )

    print(f"Loaded {len(documents)} docs (skipped {empty_count} empty).")
    return documents


if __name__ == "__main__":
    docs = load_hf_kb()
    print("First doc preview:" if docs else "No docs loaded.")
    if docs:
        print(docs[0].metadata)
        print(docs[0].page_content[:300])