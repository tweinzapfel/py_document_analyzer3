import os
import io
import json
from datetime import datetime
from typing import List, Dict, Tuple

import streamlit as st
import pandas as pd

# --- Document parsing libs ---
# PDF
import fitz  # PyMuPDF
# DOCX
from docx import Document as DocxDocument

# --- OpenAI ---
from openai import OpenAI


# -----------------------------
# Helpers: File ingestion
# -----------------------------
def extract_text_from_pdf_bytes(data: bytes) -> Tuple[str, List[int]]:
    """Return full text and a list mapping page numbers to cumulative char offsets.
    The offsets help us attribute findings to pages.
    """
    text_parts = []
    page_starts = []
    with fitz.open(stream=data, filetype="pdf") as doc:
        for i, page in enumerate(doc):
            page_starts.append(sum(len(t) for t in text_parts))
            text_parts.append(page.get_text())
    full_text = "\n".join(text_parts)
    return full_text, page_starts


def extract_text_from_docx_filelike(file_obj) -> str:
    # python-docx can read file-like objects directly
    doc = DocxDocument(file_obj)
    parts = []
    for para in doc.paragraphs:
        parts.append(para.text)
    # also include table text
    for table in doc.tables:
        for row in table.rows:
            parts.append("\t".join([cell.text for cell in row.cells]))
    return "\n".join(parts)


def chunk_text(text: str, max_chars: int = 12000, overlap: int = 600) -> List[str]:
    """Simple character-based chunking to stay under model limits.
    Adjust sizes in the sidebar if needed.
    """
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0:
            start = 0
        if start >= len(text):
            break
    return chunks


# -----------------------------
# LLM prompts & calling
# -----------------------------
SYSTEM_PROMPT = """
You are an expert contracts analyst specializing in U.S. Federal Government Requests for Proposal (RFPs) and commercial RFPs.
You read raw RFP text and produce:
1) A clear, step-by-step submission instruction checklist (with due dates, delivery portals, file formats, required forms, certifications, page limits, font/formatting requirements, questions deadlines, and evaluation criteria if present).
2) A list of key contractual risk areas with short rationale and potential mitigations (e.g., indemnification, limitation of liability, IP ownership, termination for convenience, liquidated damages, service levels, data security, insurance, most favored customer, audit rights, jurisdiction/venue, etc.).
3) Mentions of user-specified key terms/clauses (if provided) with a short excerpt and why they matter.

Output strictly as JSON with the following schema:
{
  "instructions": [
    {"step": int, "item": str, "details": str, "due_date": str|null, "submission": str|null, "format": str|null, "page_limit": str|null, "reference": {"page": int|null, "snippet": str|null}}
  ],
  "risks": [
    {"topic": str, "why_risky": str, "mitigation": str, "reference": {"page": int|null, "snippet": str|null}}
  ],
  "key_terms": [
    {"term": str, "found": bool, "context": str|null, "reference": {"page": int|null, "snippet": str|null}}
  ]
}
Keep entries concise but specific. Use ISO dates (YYYY-MM-DD) when dates are explicit; otherwise leave null.
If nothing is found for a section, return an empty array for that section.
"""


def call_openai_analyze(client: OpenAI, model: str, rfp_text: str, key_terms: List[str]) -> Dict:
    # Build a compact user message
    user_payload = {
        "key_terms": key_terms,
        "rfp_excerpt": rfp_text[:1000] + ("..." if len(rfp_text) > 1000 else ""),
        "rfp_full": rfp_text
    }

    resp = client.responses.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_payload)}
        ],
        response_format={"type": "json_object"}
    )

    # Responses API returns content in a consistent structure
    content = resp.output[0].content[0].text if hasattr(resp, 'output') else resp.choices[0].message.content
    try:
        data = json.loads(content)
    except Exception:
        # Fallback: wrap as best-effort structure
        data = {"instructions": [], "risks": [], "key_terms": []}
    return data


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="RFP Analyzer", page_icon="📄", layout="wide")

st.title("📄 RFP Analyzer")
st.caption("Upload one or more RFPs (PDF or DOCX). I’ll extract submission instructions, highlight risks, and flag your key terms. Download results as HTML or Excel.")

with st.sidebar:
    st.header("Settings")
    model = st.text_input("OpenAI model", value=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))
    chunk_size = st.number_input("Chunk size (chars)", min_value=4000, max_value=20000, value=12000, step=500)
    overlap = st.number_input("Chunk overlap (chars)", min_value=0, max_value=3000, value=600, step=100)
    st.markdown("Use **st.secrets['OPENAI_API_KEY']** or environment variable **OPENAI_API_KEY** for auth.")

key_terms_input = st.text_area(
    "Optional: key terms/clauses to flag (comma-separated)",
    help="E.g., indemnification, limitation of liability, MFC, IP ownership, data security, audit rights, termination for convenience"
)
key_terms = [t.strip() for t in key_terms_input.split(",") if t.strip()] if key_terms_input else []

uploaded_files = st.file_uploader(
    "Upload RFP files (PDF or DOCX)",
    type=["pdf", "docx"],
    accept_multiple_files=True
)

if uploaded_files:
    # Initialize OpenAI client
    client = OpenAI()

    all_results = []  # for Excel aggregation across files

    for uf in uploaded_files:
        st.subheader(f"File: {uf.name}")

        # Extract text
        ext = uf.name.lower().split(".")[-1]
        page_starts = []
        if ext == "pdf":
            data = uf.getvalue()
            raw_text, page_starts = extract_text_from_pdf_bytes(data)
        elif ext == "docx":
            raw_text = extract_text_from_docx_filelike(uf)
        else:
            st.error("Unsupported file type.")
            continue

        # Chunk and analyze
        chunks = chunk_text(raw_text, max_chars=int(chunk_size), overlap=int(overlap))
        st.caption(f"Parsed ~{len(raw_text):,} characters → {len(chunks)} chunk(s)")

        merged = {"instructions": [], "risks": [], "key_terms": []}
        for idx, ch in enumerate(chunks, start=1):
            with st.status(f"Analyzing chunk {idx}/{len(chunks)}…", expanded=False):
                result = call_openai_analyze(client, model, ch, key_terms)
                for k in merged:
                    merged[k].extend(result.get(k, []))

        # De-duplicate simple cases (by item/topic/term + snippet)
        def _dedupe(items: List[Dict], keys: List[str]) -> List[Dict]:
            seen = set()
            out = []
            for it in items:
                sig = tuple((k, str(it.get(k))) for k in keys)
                if sig not in seen:
                    seen.add(sig)
                    out.append(it)
            return out

        merged["instructions"] = _dedupe(merged["instructions"], ["item", "due_date", "format"])
        merged["risks"] = _dedupe(merged["risks"], ["topic", "why_risky"]) 
        merged["key_terms"] = _dedupe(merged["key_terms"], ["term", "context"]) 

        # Convert to DataFrames
        instr_df = pd.DataFrame(merged["instructions"]) if merged["instructions"] else pd.DataFrame(columns=["step","item","details","due_date","submission","format","page_limit","reference"])
        risks_df = pd.DataFrame(merged["risks"]) if merged["risks"] else pd.DataFrame(columns=["topic","why_risky","mitigation","reference"])
        terms_df = pd.DataFrame(merged["key_terms"]) if merged["key_terms"] else pd.DataFrame(columns=["term","found","context","reference"])

        # Sort instructions by due date then step if available
        if "due_date" in instr_df.columns:
            # Coerce dates where possible
            def _to_date(x):
                try:
                    return pd.to_datetime(x).date()
                except Exception:
                    return pd.NaT
            if not instr_df.empty:
                instr_df["due_date_parsed"] = instr_df["due_date"].apply(_to_date)
                instr_df.sort_values(["due_date_parsed","step"], inplace=True, na_position="last")
                instr_df.drop(columns=["due_date_parsed"], inplace=True)

        # Display HTML view
        with st.expander("📋 Submission Instructions (HTML)", expanded=True):
            if instr_df.empty:
                st.info("No explicit submission instructions found.")
            else:
                for i, row in instr_df.iterrows():
                    dd = row.get("due_date") or "—"
                    sub = row.get("submission") or "—"
                    fmt = row.get("format") or "—"
                    pl = row.get("page_limit") or "—"
                    ref = row.get("reference") or {}
                    ref_page = ref.get("page") if isinstance(ref, dict) else None
                    ref_snip = ref.get("snippet") if isinstance(ref, dict) else None
                    st.markdown(f"**Step {row.get('step','—')}: {row.get('item','(unspecified)')}**  ")
                    st.markdown(f"{row.get('details','')}  ")
                    st.markdown(f"**Due:** {dd} • **Submission:** {sub} • **Format:** {fmt} • **Page limit:** {pl}")
                    if ref_page or ref_snip:
                        st.caption(f"Ref page: {ref_page if ref_page is not None else '—'} | snippet: {ref_snip[:180] + ('…' if ref_snip and len(ref_snip)>180 else '')}")
                    st.markdown("---")

        with st.expander("⚠️ Risk Areas (HTML)", expanded=True):
            if risks_df.empty:
                st.info("No risk language found.")
            else:
                for i, row in risks_df.iterrows():
                    ref = row.get("reference") or {}
                    ref_page = ref.get("page") if isinstance(ref, dict) else None
                    ref_snip = ref.get("snippet") if isinstance(ref, dict) else None
                    st.markdown(f"**{row.get('topic','(unspecified)')}** — {row.get('why_risky','')}  ")
                    st.markdown(f"**Mitigation:** {row.get('mitigation','')}  ")
                    if ref_page or ref_snip:
                        st.caption(f"Ref page: {ref_page if ref_page is not None else '—'} | snippet: {ref_snip[:180] + ('…' if ref_snip and len(ref_snip)>180 else '')}")
                    st.markdown("---")

        with st.expander("🔎 Key Terms (HTML)", expanded=bool(key_terms)):
            if not key_terms:
                st.caption("No custom key terms provided.")
            elif terms_df.empty:
                st.info("No matches found for provided key terms.")
            else:
                for i, row in terms_df.iterrows():
                    ref = row.get("reference") or {}
                    ref_page = ref.get("page") if isinstance(ref, dict) else None
                    ref_snip = ref.get("snippet") if isinstance(ref, dict) else None
                    st.markdown(f"**{row.get('term')}** — found: {row.get('found')}  ")
                    if row.get("context"):
                        st.markdown(row.get("context"))
                    if ref_page or ref_snip:
                        st.caption(f"Ref page: {ref_page if ref_page is not None else '—'} | snippet: {ref_snip[:180] + ('…' if ref_snip and len(ref_snip)>180 else '')}")
                    st.markdown("---")

        # Prepare Excel
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            instr_df.to_excel(writer, sheet_name="Instructions", index=False)
            risks_df.to_excel(writer, sheet_name="Risks", index=False)
            terms_df.to_excel(writer, sheet_name="KeyTerms", index=False)
            meta_df = pd.DataFrame({
                "File": [uf.name],
                "AnalyzedAt": [datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")],
                "Model": [model],
                "Chars": [len(raw_text)],
                "Chunks": [len(chunks)]
            })
            meta_df.to_excel(writer, sheet_name="Meta", index=False)
        output.seek(0)

        st.download_button(
            label="⬇️ Download Excel report",
            data=output,
            file_name=f"{os.path.splitext(uf.name)[0]}_analysis.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

        # Also allow HTML export (simple stitched HTML)
        html_parts = [
            f"<h2>RFP Analysis — {uf.name}</h2>",
            "<h3>Submission Instructions</h3>",
            instr_df.to_html(index=False, escape=False) if not instr_df.empty else "<p>No instructions found.</p>",
            "<h3>Risk Areas</h3>",
            risks_df.to_html(index=False, escape=False) if not risks_df.empty else "<p>No risks found.</p>",
            "<h3>Key Terms</h3>",
            terms_df.to_html(index=False, escape=False) if not terms_df.empty else "<p>No key term matches found.</p>",
        ]
        html_bytes = "\n".join(html_parts).encode("utf-8")
        st.download_button(
            label="⬇️ Download HTML report",
            data=html_bytes,
            file_name=f"{os.path.splitext(uf.name)[0]}_analysis.html",
            mime="text/html",
            use_container_width=True
        )

        # Aggregate (optional multi-file workbook later)
        merged_copy = merged.copy()
        merged_copy["__file"] = uf.name
        all_results.append((uf.name, instr_df, risks_df, terms_df))

    # Optional: combined workbook across all files
    if len(all_results) > 1:
        comb = io.BytesIO()
        with pd.ExcelWriter(comb, engine="xlsxwriter") as writer:
            for fname, instr_df, risks_df, terms_df in all_results:
                instr_df.to_excel(writer, sheet_name=f"Instr_{fname[:20]}", index=False)
                risks_df.to_excel(writer, sheet_name=f"Risks_{fname[:20]}", index=False)
                terms_df.to_excel(writer, sheet_name=f"Terms_{fname[:20]}", index=False)
        comb.seek(0)
        st.download_button(
            label="⬇️ Download Combined Excel (all files)",
            data=comb,
            file_name="rfp_analysis_combined.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
else:
    st.info("Upload one or more PDF/DOCX files to begin.")


# -----------------------------
# Notes for deployment
# -----------------------------
# 1) Add your OpenAI key via Streamlit Secrets or ENV:
#    - .streamlit/secrets.toml with: OPENAI_API_KEY = "sk-..."
#    - or export OPENAI_API_KEY in the environment.
# 2) Recommended packages (requirements.txt):
#    streamlit
#    openai>=1.0.0
#    pymupdf
#    python-docx
#    pandas
#    xlsxwriter
# 3) If PDFs contain complex tables you also plan to extract structurally,
#    consider camelot or pdfplumber as a separate pass.
