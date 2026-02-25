"""
literature_search.py
--------------------
For each actionable variant from oncokb_mapping.py, search PubMed for
relevant literature and use GPT to generate a concise clinical summary.

Workflow:
  1. Read oncokb output TSV (from data/output/{case_id}_oncokb.tsv)
  2. For each gene+variant, query PubMed via Entrez API (free, no key needed)
  3. Fetch abstracts of top N papers
  4. Send to GPT-4o with a structured prompt → get 1-paragraph clinical summary
  5. Save results to data/output/{case_id}_literature.tsv

Requirements:
  pip install biopython openai
  export OPENAI_API_KEY="your_key_here"

Usage:
  python literature_search.py --input ../data/output/TCGA-49-4507_oncokb.tsv
  python literature_search.py --all   # process all *_oncokb.tsv files
"""

import os
import sys
import time
import argparse
import pandas as pd
from pathlib import Path
from typing import Optional

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
INPUT_DIR  = SCRIPT_DIR / "../data/output"
OUT_DIR    = SCRIPT_DIR / "../data/output"

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
PUBMED_EMAIL   = os.environ.get("PUBMED_EMAIL", "user@example.com")  # required by NCBI
MAX_PAPERS     = 5   # number of PubMed abstracts to fetch per variant
GPT_MODEL      = "gpt-4o"


# ── PubMed search ─────────────────────────────────────────────────────────────

def pubmed_search(gene: str, variant: str, max_results: int = MAX_PAPERS) -> list[str]:
    """
    Search PubMed for gene + variant, return list of abstracts.
    Uses Biopython Entrez (free, no API key needed, just email).
    """
    from Bio import Entrez
    Entrez.email = PUBMED_EMAIL

    query = f"{gene}[Gene] AND {variant}[Title/Abstract] AND (LUAD OR lung adenocarcinoma OR NSCLC)"
    try:
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="relevance")
        record = Entrez.read(handle)
        handle.close()
        ids = record["IdList"]
        if not ids:
            return []

        handle = Entrez.efetch(db="pubmed", id=ids, rettype="abstract", retmode="text")
        abstracts_text = handle.read()
        handle.close()

        # Split into individual abstracts (rough split by numbered entries)
        abstracts = [a.strip() for a in abstracts_text.split("\n\n\n") if a.strip()]
        return abstracts[:max_results]
    except Exception as e:
        print(f"[PubMed] Error for {gene} {variant}: {e}", file=sys.stderr)
        return []


# ── GPT summarization ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an oncology clinical assistant. Given a somatic mutation
and relevant PubMed abstracts, write a concise (3-5 sentence) clinical summary covering:
1. What is known about this mutation's role in cancer
2. Associated drugs or targeted therapies
3. Clinical evidence level (if mentioned)
Be factual and cite paper count. If no relevant abstracts, say "Limited published evidence found."
"""

def gpt_summarize(gene: str, variant: str, abstracts: list[str], oncogenic: str, treatments: str) -> str:
    """Send abstracts to GPT and get a clinical summary."""
    if not OPENAI_API_KEY:
        return "[OPENAI_API_KEY not set — skipping GPT summary]"

    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    abstracts_text = "\n\n---\n\n".join(abstracts) if abstracts else "No abstracts available."
    user_msg = (
        f"Gene: {gene}\n"
        f"Variant: {variant}\n"
        f"OncoKB Oncogenicity: {oncogenic}\n"
        f"OncoKB Treatments: {treatments}\n\n"
        f"PubMed abstracts ({len(abstracts)} papers):\n{abstracts_text}"
    )

    try:
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            max_tokens=300,
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[GPT error: {e}]"


# ── Main processing ───────────────────────────────────────────────────────────

def process_oncokb_file(oncokb_tsv: Path) -> Optional[pd.DataFrame]:
    df = pd.read_csv(oncokb_tsv, sep="\t")
    if df.empty:
        print(f"[Skip] {oncokb_tsv.name} is empty.")
        return None

    results = []
    for _, row in df.iterrows():
        gene     = str(row.get("Hugo_Symbol", ""))
        variant  = str(row.get("HGVSp_Short", ""))
        oncogenic = str(row.get("ONCOGENIC", ""))
        treatments = str(row.get("TREATMENTS", ""))
        case_id  = str(row.get("case_id", oncokb_tsv.stem.replace("_oncokb", "")))

        print(f"  Searching PubMed: {gene} {variant} ...")
        abstracts = pubmed_search(gene, variant)
        time.sleep(0.5)  # NCBI rate limit: max 3 req/sec without API key

        print(f"  GPT summarizing ({len(abstracts)} papers) ...")
        summary = gpt_summarize(gene, variant, abstracts, oncogenic, treatments)

        results.append({
            "case_id":       case_id,
            "gene":          gene,
            "variant":       variant,
            "oncogenic":     oncogenic,
            "treatments":    treatments,
            "pubmed_count":  len(abstracts),
            "clinical_summary": summary,
        })

    result_df = pd.DataFrame(results)
    out_path = OUT_DIR / oncokb_tsv.name.replace("_oncokb.tsv", "_literature.tsv")
    result_df.to_csv(out_path, sep="\t", index=False)
    print(f"[Done] {len(result_df)} variants → {out_path}")
    return result_df


def main():
    parser = argparse.ArgumentParser(description="PubMed + GPT literature search for LUAD variants")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=Path, help="Path to a single *_oncokb.tsv file")
    group.add_argument("--all",   action="store_true", help="Process all *_oncokb.tsv in data/output/")
    args = parser.parse_args()

    if args.input:
        process_oncokb_file(args.input)
    else:
        files = sorted(INPUT_DIR.glob("*_oncokb.tsv"))
        if not files:
            print(f"[ERROR] No *_oncokb.tsv files found in {INPUT_DIR}", file=sys.stderr)
            sys.exit(1)
        all_results = []
        for f in files:
            print(f"\n=== Processing {f.name} ===")
            r = process_oncokb_file(f)
            if r is not None:
                all_results.append(r)
        if all_results:
            combined = pd.concat(all_results, ignore_index=True)
            combined.to_csv(OUT_DIR / "all_samples_literature.tsv", sep="\t", index=False)
            print(f"\n[Summary] Combined → {OUT_DIR / 'all_samples_literature.tsv'}")


if __name__ == "__main__":
    main()
