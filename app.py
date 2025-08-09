"""
Streamlit Project Management Assistant App
----------------------------------------

This Streamlit application is designed for project managers who want to
consolidate all of their project documentation, track progress, and
receive intelligent insights via the OpenAI API.  Upload your project
scope and schedule documents, log team updates, and interact with an
AI‑powered chat assistant to surface what has been accomplished,
identify outstanding tasks, and brainstorm next steps.

To use this application you'll need an OpenAI API key.  Place the key
in your Streamlit secrets file under the variable ``OPENAI_API_KEY``.
Example ``.streamlit/secrets.toml`` content::

    OPENAI_API_KEY = "sk‑..."

Run the app with ``streamlit run app.py``.  This file should live in the
root of your Streamlit project.
"""

from __future__ import annotations

import io
import os
import json
from typing import List, Dict, Any

import streamlit as st

try:
    import openai  # type: ignore
except ImportError:
    openai = None

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None


def _ensure_openai_client() -> bool:
    if openai is None:
        st.error("openai package is not installed.")
        return False
    api_key = st.secrets.get("OPENAI_API_KEY", None)
    if not api_key:
        st.info("Add your OpenAI API key in .streamlit/secrets.toml as 'OPENAI_API_KEY'.")
        return False
    openai.api_key = api_key
    return True


def _call_chat_completion(messages: List[Dict[str, Any]], model: str = "gpt-4", temperature: float = 0.3) -> Any:
    if not _ensure_openai_client():
        raise RuntimeError("OpenAI client is not ready.")
    if hasattr(openai, "OpenAI"):
        client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        return client.chat.completions.create(model=model, messages=messages, temperature=temperature)
    return openai.ChatCompletion.create(model=model, messages=messages, temperature=temperature)


def _extract_text_from_pdf(file_bytes: bytes) -> str:
    if fitz is None:
        return ""
    try:
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            return "\n".join([page.get_text() for page in doc])
    except Exception as exc:
        st.error(f"Failed to parse PDF: {exc}")
        return ""


def _read_uploaded_file(uploaded_file: Any) -> str:
    if uploaded_file is None:
        return ""
    name = uploaded_file.name.lower()
    if name.endswith((".txt", ".md", ".csv", ".json", ".yaml", ".yml")):
        return uploaded_file.read().decode("utf-8", errors="ignore")
    elif name.endswith(".pdf"):
        return _extract_text_from_pdf(uploaded_file.read())
    else:
        st.warning("Unsupported file type.")
        return ""


def _rewrite_summary_state_with_gpt(context: str) -> None:
    """
    Calls the model to rewrite the structured summary JSON in session state,
    then renders a readable Markdown progress summary. Robust to dicts/lists.
    """
    import re

    instructions = (
        "You are a project assistant. Maintain a concise, structured project summary as JSON. "
        "Update, merge, or remove sections as appropriate based on the provided context. "
        "Keep keys stable when possible (e.g., goals, milestones, risks, blockers, decisions, next_steps). "
        "Return JSON only—no prose, no code fences."
    )

    previous = st.session_state.get("summary_state", {}) or {}

    prompt = f"""
Current summary (JSON):
{json.dumps(previous, ensure_ascii=False, indent=2)}

Context:
{context}

Return ONLY the updated summary as valid JSON (object). No commentary.
"""

    messages = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": prompt},
    ]

    # --- Call model and parse JSON safely ---
    try:
        response = _call_chat_completion(messages)
        # Handle both possible SDK shapes
        if hasattr(response, "choices"):
            content = response.choices[0].message.content
        else:
            content = response["choices"][0]["message"]["content"]
        content = (content or "").strip()

        # If the model included code fences or extra text, try to extract the JSON object
        # by grabbing the largest {...} block.
        match = re.search(r"\{[\s\S]*\}", content)
        if match:
            content = match.group(0)

        updated = json.loads(content)
        if not isinstance(updated, dict):
            raise ValueError("Model returned JSON that is not an object.")
    except Exception as e:
        st.warning(f"Could not update summary: {e}")
        return

    # --- Save structured state ---
    st.session_state.summary_state = updated

    # --- Render Markdown summary robustly (handles dicts/lists/values, recursively) ---
    def to_text(value: Any) -> str:
        """Convert any value to a single-line string."""
        if value is None:
            return ""
        if isinstance(value, (str, int, float, bool)):
            return str(value)
        try:
            # Compact JSON for nested structures
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            return str(value)

    def sanitize(s: str) -> str:
        return s.replace("\n", " ").strip()

    def render_markdown(value: Any, level: int = 0) -> list[str]:
        """Recursively render dict/list/value to a list of Markdown lines."""
        lines: list[str] = []
        indent = "  " * max(level, 0)

        if isinstance(value, dict):
            for k, v in value.items():
                if isinstance(v, (dict, list)):
                    # Key as bold section header
                    lines.append(f"{indent}**{k}**")
                    lines.extend(render_markdown(v, level + 1))
                else:
                    lines.append(f"{indent}**{k}**: {sanitize(to_text(v))}")
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, (dict, list)):
                    lines.append(f"{indent}-")
                    lines.extend(render_markdown(item, level + 1))
                else:
                    lines.append(f"{indent}- {sanitize(to_text(item))}")
        else:
            lines.append(f"{indent}{sanitize(to_text(value))}")

        return lines

    md_lines = render_markdown(updated, level=0)
    st.session_state.progress = "\n".join(md_lines) if md_lines else ""



def _generate_chat_reply(messages: List[Dict[str, str]], context: str) -> str:
    preamble = [
        {"role": "system", "content": (
            "You are a helpful assistant for project managers. Respond using the user's documents and summary."
        )},
        {"role": "system", "content": f"Project context:\n{context}"},
    ] + messages
    try:
        response = _call_chat_completion(preamble)
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Chat error: {e}")
        return ""


def main():
    st.set_page_config(page_title="Project Assistant", layout="wide")
    st.title("📋 Project Management Assistant")
    st.markdown("Upload project-related documents and chat to maintain a dynamic summary.")

    for k, v in [
        ("messages", []),
        ("uploaded_files", []),
        ("progress", ""),
        ("summary_state", {}),
    ]:
        if k not in st.session_state:
            st.session_state[k] = v

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("📁 Upload Files")
        uploads = st.file_uploader("Drop project files (scope, schedule, notes, etc.)", type=["pdf", "txt", "md", "csv", "json", "yaml", "yml"], accept_multiple_files=True)
        if uploads:
            for file in uploads:
                text = _read_uploaded_file(file)
                if text:
                    st.session_state.uploaded_files.append((file.name, text))
                    _rewrite_summary_state_with_gpt(text)
            st.success(f"Uploaded {len(uploads)} file(s).")

        if st.session_state.uploaded_files:
            st.markdown("#### Files Received:")
            for fname, _ in st.session_state.uploaded_files:
                st.markdown(f"- {fname}")

        st.subheader("💬 Chat with the Assistant")
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_input = st.chat_input("Ask a question or update project status...")
        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            _rewrite_summary_state_with_gpt(user_input)

            context = "\n\n".join([t for _, t in st.session_state.uploaded_files])
            reply = _generate_chat_reply(messages=st.session_state.messages, context=context)
            if reply:
                st.session_state.messages.append({"role": "assistant", "content": reply})
                _rewrite_summary_state_with_gpt(reply)

    with col2:
        st.subheader("📌 Progress Summary")
        if st.session_state.progress:
            st.markdown(st.session_state.progress)
        else:
            st.info("Upload documents or chat to generate a project summary.")


if __name__ == "__main__":
    main()
