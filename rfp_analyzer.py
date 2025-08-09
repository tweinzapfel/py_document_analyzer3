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

BASE_SYSTEM_PROMPT = """
You are an expert contracts analyst specializing in Requests for Proposal (RFPs).
You read raw RFP text and produce:
1) A clear, step-by-step submission instruction checklist (with due dates, delivery portals, file formats, required forms, certifications, page limits, font/formatting requirements, questions deadlines, and evaluation criteria if present).
2) A list of key contractual risk areas with short rationale and potential mitigations.
3) Mentions of user-specified key terms/clauses (if provided) with a short excerpt and why they matter.

Context profile: <<CONTEXT_PROFILE>>
- If Government (U.S. Federal), focus on FAR/DFARS references, proposal volumes, forms (SF 1449/33, etc.), representations/certifications (SAM, small business, Section 889), submission portals (SAM, PIEE), compliance dates, and typical Gov risks (T4C, data rights, MFC, audit).
- If Commercial/Non-government, focus on indemnities, limitation of liability caps, IP ownership, SLAs/LDs, payment terms, data security/privacy, insurance, jurisdiction/venue.

Output strictly as JSON with the schema: {"instructions":[],"risks":[],"key_terms":[]}
Use ISO dates when explicit; otherwise null. Keep entries concise but specific.
"""

def build_system_prompt(profile: str) -> str:
    profile_note = "Government (U.S. Federal)" if profile.startswith("Government") else "Commercial / Non-government"
    return BASE_SYSTEM_PROMPT.replace("<<CONTEXT_PROFILE>>", profile_note)

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

# The rest of the app code remains unchanged from your last version
