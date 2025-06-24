import pandas as pd
from pathlib import Path
import json, re, textwrap, html

RAW = Path("annotated-20250618T233549Z-1-001/annotated/unified_error_dataset/20250616_192824/unified_error_dataset_annotated.csv")
OUT = Path("data/scam_detection.jsonl")

def strip_html(text: str) -> str:
    # crude but effective for most email bodies
    text = html.unescape(re.sub(r"<[^>]+>", "", text))
    # normalise whitespace & trim
    return re.sub(r"\s+", " ", text).strip()

def build_record(row) -> dict:
    content   = strip_html(row["original_content"])
    label   = row["class"].strip().title()   # Scam or Legitimate
    expl    = row["explanation"].strip()
    keys    = row["key_indicators"].strip()

    # --- prompt / answer formulation ---
    system  = "You are an AI assistant specialised in detecting scam " \
              "or legitimate content."
    user    = textwrap.dedent(f"""
        Classify the following content **strictly** as `Scam` or `Legitimate`
        and briefly justify your answer in one sentence.

        <CONTENT>
        {content}
        </CONTENT>
    """).strip()

    assistant = f"{label}\nReason: {expl}"

    # Qwen-3 chat template
    text = (
        "<|system|>\n"   + system   + "\n"
        "<|user|>\n"     + user     + "\n"
        "<|assistant|>\n"+ assistant
    )

    return {"text": text}

def main():
    df = pd.read_csv(RAW)

    # 1️⃣ basic hygiene
    df = df.drop_duplicates(subset="original_content") \
           .dropna(subset=["original_content", "class", "explanation"])

    # 2️⃣ standardise numeric label
    df["class"] = df["class"].replace({"0": "Legitimate", "1": "Scam"})

    # 3️⃣ build chat-formatted rows
    records = [build_record(r) for _, r in df.iterrows()]

    # 4️⃣ write JSONL
    OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records))
    print(f"Wrote {len(records):,} records → {OUT}")

if __name__ == "__main__":
    main()