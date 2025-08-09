import os
import io
import json
from datetime import datetime
from typing import List, Dict, Tuple

import streamlit as st
import pandas as pd
import fitz  # PyMuPDF
from docx import Document as DocxDocument
from openai import OpenAI

def extract_text_from_pdf_bytes(data: bytes) -> Tuple[str, List[int]]:
    text_parts = []
    page_starts = []
    with fitz.open(stream=data, filetype="pdf") as doc:
        for i, page in enumerate(doc):
            page_starts.append(sum(len(t) for t in text_parts))
            text_parts.append(page.get_text())
    full_text = "\n".join(text_parts)
    return full_text, page_starts

def extract_text_from_docx_bytes(data: bytes) -> str:
    doc = DocxDocument(io.BytesIO(data))
    parts = []
    for para in doc.paragraphs:
        parts.append(para.text)
    for table in doc.tables:
        for row in table.rows:
            parts.append("\t".join([cell.text for cell in row.cells]))
    return "\n".join(parts)

def chunk_text(text: str, max_chars: int = 12000, overlap: int = 600) -> List[str]:
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

SYSTEM_PROMPT = """
You are an expert contracts analyst specializing in U.S. Federal Government Requests for Proposal (RFPs) and commercial RFPs.
You read raw RFP text and produce:
1) A clear, step-by-step submission instruction checklist (with due dates, delivery portals, file formats, required forms, certifications, page limits, font/formatting requirements, questions deadlines, and evaluation criteria if present).
2) A list of key contractual risk areas with short rationale and potential mitigations.
3) Mentions of user-specified key terms/clauses (if provided) with a short excerpt and why they matter.

Output strictly as JSON with the schema: {"instructions":[],"risks":[],"key_terms":[]}
"""

def call_openai_analyze(client: OpenAI, model: str, rfp_text: str, key_terms: List[str]) -> Dict:
    user_payload = {
        "key_terms": key_terms,
        "rfp_excerpt": rfp_text[:1000] + ("..." if len(rfp_text) > 1000 else ""),
        "rfp_full": rfp_text
    }
    try:
        resp = client.responses.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(user_payload)}
            ],
            response_format={"type": "json_object"}
        )
        if hasattr(resp, "output_text") and resp.output_text:
            content = resp.output_text
        elif hasattr(resp, "output") and resp.output:
            content = resp.output[0].content[0].text
        else:
            content = resp.choices[0].message.content
    except Exception:
        chat = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(user_payload)}
            ],
            response_format={"type": "json_object"}
        )
        content = chat.choices[0].message.content
    try:
        data = json.loads(content)
    except Exception:
        data = {"instructions": [], "risks": [], "key_terms": []}
    return data

st.set_page_config(page_title="RFP Analyzer", page_icon="📄", layout="wide")

st.title("📄 RFP Analyzer")
st.caption("Upload one or more RFPs (PDF or DOCX) to extract submission instructions, highlight risks, and flag your key terms.")

with st.sidebar:
    st.header("Settings")
    model = st.text_input("OpenAI model", value=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    chunk_size = st.number_input("Chunk size (chars)", min_value=4000, max_value=20000, value=12000, step=500)
    overlap = st.number_input("Chunk overlap (chars)", min_value=0, max_value=3000, value=600, step=100)
    diagnostics = st.toggle("Diagnostics mode", value=False)
    st.markdown("Use **st.secrets['OPENAI_API_KEY']** or environment variable **OPENAI_API_KEY** for auth.")

key_terms_input = st.text_area("Optional: key terms/clauses to flag (comma-separated)")
key_terms = [t.strip() for t in key_terms_input.split(",") if t.strip()] if key_terms_input else []

uploaded_files = st.file_uploader("Upload RFP files (PDF or DOCX)", type=["pdf", "docx"], accept_multiple_files=True)

if uploaded_files:
    api_key = None
    try:
        api_key = st.secrets.get("OPENAI_API_KEY")  # type: ignore
    except Exception:
        api_key = None
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("No OpenAI API key found.")
    client = OpenAI(api_key=api_key)

    for uf in uploaded_files:
        try:
            st.subheader(f"File: {uf.name}")
            ext = uf.name.lower().split(".")[-1]
            if ext == "pdf":
                data = uf.getvalue()
                raw_text, _ = extract_text_from_pdf_bytes(data)
            elif ext == "docx":
                data = uf.getvalue()
                raw_text = extract_text_from_docx_bytes(data)
            else:
                st.error("Unsupported file type.")
                continue

            chunks = chunk_text(raw_text, max_chars=int(chunk_size), overlap=int(overlap))
            st.caption(f"Parsed ~{len(raw_text):,} characters → {len(chunks)} chunk(s)")

            merged = {"instructions": [], "risks": [], "key_terms": []}
            for idx, ch in enumerate(chunks, start=1):
                try:
                    with st.status(f"Analyzing chunk {idx}/{len(chunks)}…"):
                        result = call_openai_analyze(client, model, ch, key_terms)
                        for k in merged:
                            merged[k].extend(result.get(k, []))
                except Exception as e:
                    if diagnostics:
                        import traceback
                        st.exception(e)
                        st.text(traceback.format_exc())
                    else:
                        st.error(f"LLM analysis failed on chunk {idx}.")

            instr_df = pd.DataFrame(merged["instructions"])
            risks_df = pd.DataFrame(merged["risks"])
            terms_df = pd.DataFrame(merged["key_terms"])

            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                instr_df.to_excel(writer, sheet_name="Instructions", index=False)
                risks_df.to_excel(writer, sheet_name="Risks", index=False)
                terms_df.to_excel(writer, sheet_name="KeyTerms", index=False)
            output.seek(0)

            st.download_button("⬇️ Download Excel report", output, f"{os.path.splitext(uf.name)[0]}_analysis.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except Exception as e:
            if diagnostics:
                import traceback
                st.exception(e)
                st.text(traceback.format_exc())
            else:
                st.error(f"Something went wrong processing {uf.name}.")
else:
    st.info("Upload one or more PDF/DOCX files to begin.")
