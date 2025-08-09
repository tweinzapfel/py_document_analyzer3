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

# -----------------------------
# Extraction helpers
# -----------------------------

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

# -----------------------------
# Prompting
# -----------------------------

BASE_SYSTEM_PROMPT = """
You are an expert contracts analyst specializing in Requests for Proposal (RFPs).
You read raw RFP text and produce:
1) A clear, step-by-step submission instruction checklist (with due dates, delivery portals, file formats, required forms, certifications, page limits, font/formatting requirements, questions deadlines, and evaluation criteria if present).
2) A list of key contractual risk areas with short rationale and potential mitigations.
3) Mentions of user-specified key terms/clauses (if provided) with a short excerpt and why they matter.

Context profile: {context_profile}
- If Government (U.S. Federal), focus on FAR/DFARS references, proposal volumes, forms (SF 1449/33, etc.), representations/certifications (SAM, small business, Section 889), submission portals (SAM, PIEE), compliance dates, and typical Gov risks (T4C, data rights, MFC, audit).
- If Commercial/Non-government, focus on indemnities, limitation of liability caps, IP ownership, SLAs/LDs, payment terms, data security/privacy, insurance, jurisdiction/venue.

Output strictly as JSON with the schema: {"instructions":[],"risks":[],"key_terms":[]}
Use ISO dates when explicit; otherwise null. Keep entries concise but specific.
"""


def build_system_prompt(profile: str) -> str:
    profile_note = "Government (U.S. Federal)" if profile.startswith("Government") else "Commercial / Non-government"
    return BASE_SYSTEM_PROMPT.format(context_profile=profile_note)


# -----------------------------
# OpenAI call
# -----------------------------

def call_openai_analyze(client: OpenAI, model: str, system_prompt: str, rfp_text: str, key_terms: List[str]) -> Dict:
    user_payload = {
        "key_terms": key_terms,
        "rfp_excerpt": rfp_text[:1000] + ("..." if len(rfp_text) > 1000 else ""),
        "rfp_full": rfp_text
    }
    try:
        resp = client.responses.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
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
                {"role": "system", "content": system_prompt},
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

# -----------------------------
# UI
# -----------------------------

st.set_page_config(page_title="RFP Analyzer", page_icon="📄", layout="wide")

st.title("📄 RFP Analyzer")
st.caption("Queue multiple RFPs (PDF/DOCX), then analyze them in one go. Get instructions, risks, and key-term flags. View HTML or download combined Excel.")

# Session state for batching files & widget reset
if "batch_files" not in st.session_state:
    st.session_state.batch_files = []  # list of dicts {name, type, data}
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = "uploader-1"

with st.sidebar:
    st.header("Settings")
    model = st.text_input("OpenAI model", value=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    profile = st.radio(
        "RFP profile",
        ["Government (U.S. Federal)", "Commercial / Non-government"],
        help="Tailors the analysis toward FAR/DFARS & federal nuances vs general commercial terms",
    )
    chunk_size = st.number_input("Chunk size (chars)", min_value=4000, max_value=20000, value=12000, step=500)
    overlap = st.number_input("Chunk overlap (chars)", min_value=0, max_value=3000, value=600, step=100)
    diagnostics = st.toggle("Diagnostics mode", value=False, help="Show tracebacks on errors")
    st.markdown("Use **st.secrets['OPENAI_API_KEY']** or env **OPENAI_API_KEY** for auth.")

key_terms_input = st.text_area("Optional: key terms/clauses to flag (comma-separated)")
key_terms = [t.strip() for t in key_terms_input.split(",") if t.strip()] if key_terms_input else []

# --- File Staging ---
st.subheader("1) Add files to your batch")
uploaded_files = st.file_uploader(
    "Upload RFP files (PDF or DOCX)",
    type=["pdf", "docx"],
    accept_multiple_files=True,
    key=st.session_state.uploader_key,
)
col_add, col_clear = st.columns([1,1])
with col_add:
    if st.button("➕ Add selected to batch", use_container_width=True):
        if uploaded_files:
            for uf in uploaded_files:
                st.session_state.batch_files.append({
                    "name": uf.name,
                    "type": uf.name.lower().split(".")[-1],
                    "data": uf.getvalue(),
                })
            # reset uploader by changing its key
            st.session_state.uploader_key = f"uploader-{len(st.session_state.batch_files)}"
            st.rerun()
with col_clear:
    if st.button("🗑️ Clear batch", use_container_width=True):
        st.session_state.batch_files = []
        st.session_state.uploader_key = "uploader-1"
        st.rerun()

if st.session_state.batch_files:
    st.success(f"Queued {len(st.session_state.batch_files)} file(s)")
    st.dataframe(pd.DataFrame([{"File": f["name"], "Type": f["type"], "Size (KB)": round(len(f["data"]) / 1024, 1)} for f in st.session_state.batch_files]))
else:
    st.info("No files in batch yet. Upload and click 'Add selected to batch'.")

# --- Analyze Button ---
st.subheader("2) Analyze")
run_analysis = st.button("▶️ Analyze batch", type="primary", use_container_width=True, disabled=not st.session_state.batch_files)

if run_analysis:
    # Setup OpenAI client
    api_key = None
    try:
        api_key = st.secrets.get("OPENAI_API_KEY")  # type: ignore
    except Exception:
        api_key = None
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("No OpenAI API key found.")
        st.stop()

    client = OpenAI(api_key=api_key)
    system_prompt = build_system_prompt(profile)

    # Collect combined results across files
    combined_instr = []
    combined_risks = []
    combined_terms = []

    combined_html_parts = ["<h1>RFP Combined Analysis</h1>"]

    for f in st.session_state.batch_files:
        try:
            st.subheader(f"File: {f['name']}")
            ext = f["type"]
            if ext == "pdf":
                raw_text, _ = extract_text_from_pdf_bytes(f["data"])
            elif ext == "docx":
                raw_text = extract_text_from_docx_bytes(f["data"]) 
            else:
                st.error("Unsupported file type.")
                continue

            chunks = chunk_text(raw_text, max_chars=int(chunk_size), overlap=int(overlap))
            st.caption(f"Parsed ~{len(raw_text):,} characters → {len(chunks)} chunk(s)")

            merged = {"instructions": [], "risks": [], "key_terms": []}
            for idx, ch in enumerate(chunks, start=1):
                try:
                    with st.status(f"Analyzing chunk {idx}/{len(chunks)}…"):
                        result = call_openai_analyze(client, model, system_prompt, ch, key_terms)
                        for k in merged:
                            merged[k].extend(result.get(k, []))
                except Exception as e:
                    if diagnostics:
                        import traceback
                        st.exception(e)
                        st.text(traceback.format_exc())
                    else:
                        st.error(f"LLM analysis failed on chunk {idx}.")

            # DataFrames (per file)
            instr_df = pd.DataFrame(merged["instructions"]) if merged["instructions"] else pd.DataFrame(columns=["step","item","details","due_date","submission","format","page_limit","reference"]) 
            risks_df = pd.DataFrame(merged["risks"]) if merged["risks"] else pd.DataFrame(columns=["topic","why_risky","mitigation","reference"]) 
            terms_df = pd.DataFrame(merged["key_terms"]) if merged["key_terms"] else pd.DataFrame(columns=["term","found","context","reference"]) 

            # Add File column for combined export
            for df in (instr_df, risks_df, terms_df):
                if not df.empty:
                    df.insert(0, "File", f['name'])

            # Append to combined lists
            if not instr_df.empty:
                combined_instr.append(instr_df)
            if not risks_df.empty:
                combined_risks.append(risks_df)
            if not terms_df.empty:
                combined_terms.append(terms_df)

            # Per-file HTML view + download
            with st.expander("📄 View results (HTML)", expanded=True):
                parts = [
                    f"<h2>{f['name']}</h2>",
                    "<h3>Submission Instructions</h3>",
                    instr_df.to_html(index=False, escape=False) if not instr_df.empty else "<p>No instructions found.</p>",
                    "<h3>Risk Areas</h3>",
                    risks_df.to_html(index=False, escape=False) if not risks_df.empty else "<p>No risks found.</p>",
                    "<h3>Key Terms</h3>",
                    terms_df.to_html(index=False, escape=False) if not terms_df.empty else "<p>No key term matches found.</p>",
                ]
                html_bytes = "\n".join(parts).encode("utf-8")
                st.download_button(
                    label="⬇️ Download HTML (this file)",
                    data=html_bytes,
                    file_name=f"{os.path.splitext(f['name'])[0]}_analysis.html",
                    mime="text/html",
                    use_container_width=True,
                )
                # Also add to combined HTML
                combined_html_parts.extend(parts)

            # Export Excel per file
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                instr_df.to_excel(writer, sheet_name="Instructions", index=False)
                risks_df.to_excel(writer, sheet_name="Risks", index=False)
                terms_df.to_excel(writer, sheet_name="KeyTerms", index=False)
                meta_df = pd.DataFrame({
                    "File": [f['name']],
                    "AnalyzedAt": [datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")],
                    "Model": [model],
                    "Profile": [profile],
                    "Chars": [len(raw_text)],
                    "Chunks": [len(chunks)]
                })
                meta_df.to_excel(writer, sheet_name="Meta", index=False)
            output.seek(0)

            st.download_button(
                "⬇️ Download Excel (this file)",
                output,
                f"{os.path.splitext(f['name'])[0]}_analysis.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        except Exception as e:
            if diagnostics:
                import traceback
                st.exception(e)
                st.text(traceback.format_exc())
            else:
                st.error(f"Something went wrong processing {f['name']}.")

    # --- Combined exports across all files ---
    if combined_instr or combined_risks or combined_terms:
        st.markdown("---")
        st.subheader("📦 Combined results (all files)")

        comb_instr_df = pd.concat(combined_instr, ignore_index=True) if combined_instr else pd.DataFrame(columns=["File","step","item","details","due_date","submission","format","page_limit","reference"]) 
        comb_risks_df = pd.concat(combined_risks, ignore_index=True) if combined_risks else pd.DataFrame(columns=["File","topic","why_risky","mitigation","reference"]) 
        comb_terms_df = pd.concat(combined_terms, ignore_index=True) if combined_terms else pd.DataFrame(columns=["File","term","found","context","reference"]) 

        # Show a quick peek table (limited rows) to avoid giant render
        with st.expander("Preview combined tables", expanded=False):
            st.write("Instructions (first 200 rows)")
            st.dataframe(comb_instr_df.head(200))
            st.write("Risks (first 200 rows)")
            st.dataframe(comb_risks_df.head(200))
            st.write("Key Terms (first 200 rows)")
            st.dataframe(comb_terms_df.head(200))

        # Combined Excel
        comb_xlsx = io.BytesIO()
        with pd.ExcelWriter(comb_xlsx, engine="xlsxwriter") as writer:
            comb_instr_df.to_excel(writer, sheet_name="Instructions_All", index=False)
            comb_risks_df.to_excel(writer, sheet_name="Risks_All", index=False)
            comb_terms_df.to_excel(writer, sheet_name="KeyTerms_All", index=False)
            meta = pd.DataFrame({
                "GeneratedAt": [datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")],
                "Model": [model],
                "Profile": [profile],
                "Files": [len(st.session_state.batch_files)]
            })
            meta.to_excel(writer, sheet_name="Meta", index=False)
        comb_xlsx.seek(0)
        st.download_button(
            "⬇️ Download Combined Excel (all files)",
            comb_xlsx,
            "rfp_analysis_combined.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

        # Combined HTML (stitched)
        comb_html_bytes = "\n".join(combined_html_parts).encode("utf-8")
        with st.expander("🌐 View combined HTML", expanded=False):
            st.components.v1.html("\n".join(combined_html_parts), height=600, scrolling=True)
        st.download_button(
            "⬇️ Download Combined HTML",
            comb_html_bytes,
            "rfp_analysis_combined.html",
            "text/html",
            use_container_width=True,
        )

# Footer
st.markdown("---")
st.caption("Tip: You can queue files in several rounds, then click Analyze once.")
