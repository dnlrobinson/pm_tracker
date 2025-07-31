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
#this is hardd...

from __future__ import annotations

import io
import os
from typing import List, Dict, Any

import streamlit as st

try:
    import openai  # type: ignore
except ImportError:
    openai = None

try:
    import fitz  # PyMuPDF for PDF text extraction
except ImportError:
    fitz = None


def _extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from a PDF file using PyMuPDF.

    Parameters
    ----------
    file_bytes : bytes
        The raw bytes of the uploaded PDF file.

    Returns
    -------
    str
        Concatenated text of all pages in the PDF.

    Notes
    -----
    If PyMuPDF is not installed, an empty string is returned.
    """
    if fitz is None:
        return ""
    text = []
    try:
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            for page in doc:
                text.append(page.get_text())
    except Exception as exc:  # pragma: no cover - best effort to decode
        st.error(f"Failed to parse PDF: {exc}")
        return ""
    return "\n".join(text)


def _read_uploaded_file(uploaded_file: Any) -> str:
    """Read text content from an uploaded file.

    Parameters
    ----------
    uploaded_file : UploadedFile
        The file object provided by Streamlit's file uploader.

    Returns
    -------
    str
        Extracted text from the file.  Supported formats include text,
        markdown, CSV, and PDF.  Unsupported formats return an empty
        string with an error message displayed in the Streamlit app.
    """
    if uploaded_file is None:
        return ""
    file_name = uploaded_file.name.lower()
    # Handle plain text, markdown, and CSV directly
    if file_name.endswith((".txt", ".md", ".csv", ".json", ".yaml", ".yml")):
        try:
            content_bytes = uploaded_file.read()
            return content_bytes.decode("utf-8", errors="ignore")
        except Exception as exc:  # pragma: no cover
            st.error(f"Failed to read file: {exc}")
            return ""
    # Handle PDF via PyMuPDF
    elif file_name.endswith(".pdf"):
        file_bytes = uploaded_file.read()
        return _extract_text_from_pdf(file_bytes)
    else:
        st.warning(f"Unsupported file type: {file_name}. Please upload text or PDF files.")
        return ""


def _ensure_openai_client() -> bool:
    """Ensure the OpenAI client is configured and available.

    Returns
    -------
    bool
        True if OpenAI client can be used, False otherwise.
    """
    if openai is None:
        st.error("openai package is not installed. Please install it to use the chat feature.")
        return False
    api_key = st.secrets.get("OPENAI_API_KEY", None)
    if not api_key:
        st.info("Please add your OpenAI API key to the Streamlit secrets file under 'OPENAI_API_KEY'.")
        return False
    openai.api_key = api_key
    return True


def _generate_summary(scope: str, schedule: str, updates: str) -> str:
    """Generate a summary of the project status using the OpenAI API.

    This function composes a prompt that instructs the language model to
    return three sections: tasks completed, tasks pending, and
    suggestions for next steps.  The input incorporates the project
    scope, the schedule, and all team updates recorded so far.

    Parameters
    ----------
    scope : str
        The text of the project scope document.
    schedule : str
        The text of the project schedule document.
    updates : str
        Cumulative updates and transcripts provided by the team.

    Returns
    -------
    str
        A summarised report suitable for display in the UI.  If the
        OpenAI client is unavailable or an error occurs, an empty
        string is returned.
    """
    if not _ensure_openai_client():
        return ""
    summarization_prompt = (
        "You are a helpful project management assistant. Based on the "
        "provided project scope, schedule, and team updates, summarise "
        "the current state of the project. Break your answer into three "
        "clear sections: (1) What has been accomplished, (2) What still "
        "needs to be done, and (3) Suggestions or risks the team should "
        "consider next. Use short paragraphs or bullet points where "
        "appropriate.\n\n"
        f"Project scope:\n{scope}\n\n"
        f"Project schedule:\n{schedule}\n\n"
        f"Team updates:\n{updates}"
    )
    messages = [
        {"role": "system", "content": "You are a professional project management assistant."},
        {"role": "user", "content": summarization_prompt},
    ]
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.3,
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as exc:  # pragma: no cover
        st.error(f"Error generating summary: {exc}")
        return ""


def _generate_chat_reply(messages: List[Dict[str, str]], scope: str, schedule: str, updates: str) -> str:
    """Generate a chat reply from the assistant using conversation history.

    Parameters
    ----------
    messages : list of dict
        Conversation history stored in st.session_state["messages"].  Each
        element is a dictionary with keys ``role`` and ``content``.
    scope : str
        The text of the project scope document.
    schedule : str
        The text of the project schedule document.
    updates : str
        Cumulative updates and transcripts provided by the team.

    Returns
    -------
    str
        The assistant's reply.  If the OpenAI client is unavailable or an
        error occurs, an empty string is returned.
    """
    if not _ensure_openai_client():
        return ""
    # Prepend context to the chat: include project scope, schedule and updates
    conversation: List[Dict[str, str]] = []
    # Use a system message to instruct the model
    conversation.append({"role": "system", "content": (
        "You are a helpful assistant that answers questions and provides guidance "
        "related to a project's scope, schedule, and progress updates. "
        "Always reference the provided context when formulating answers."
    )})
    # Provide context in hidden system messages
    conversation.append({"role": "system", "content": f"Project scope:\n{scope}"})
    conversation.append({"role": "system", "content": f"Project schedule:\n{schedule}"})
    conversation.append({"role": "system", "content": f"Team updates:\n{updates}"})
    # Append the conversation history
    conversation.extend(messages)
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=conversation,
            temperature=0.4,
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as exc:  # pragma: no cover
        st.error(f"Error during chat completion: {exc}")
        return ""


def main() -> None:
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="Project Assistant", layout="wide")
    st.title("📋 Project Management Assistant")
    st.markdown(
        """
        Upload your project scope and schedule, log updates from your team, and
        chat with an AI assistant to surface progress summaries.  The right
        panel automatically summarises what has been done, what remains, and
        offers suggestions based on the documents and updates you provide.
        """
    )

    # Initialise session state variables
    for key, default in [
        ("messages", []),
        ("scope_text", ""),
        ("schedule_text", ""),
        ("updates", ""),
        ("progress", ""),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    # Layout: two columns separated by a vertical rule
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.subheader("Upload Documents")
        # Project scope uploader
        scope_file = st.file_uploader(
            "Project Scope (PDF/Text/CSV)", type=["pdf", "txt", "md", "csv", "json", "yaml", "yml"], key="scope_uploader"
        )
        if scope_file is not None:
            st.session_state.scope_text = _read_uploaded_file(scope_file)
            st.success("Project scope uploaded and parsed successfully.")

        # Project schedule uploader
        schedule_file = st.file_uploader(
            "Project Schedule (PDF/Text/CSV)", type=["pdf", "txt", "md", "csv", "json", "yaml", "yml"], key="schedule_uploader"
        )
        if schedule_file is not None:
            st.session_state.schedule_text = _read_uploaded_file(schedule_file)
            st.success("Project schedule uploaded and parsed successfully.")

        st.divider()
        st.subheader("Record Team Updates")
        update_text = st.text_area(
            "Describe what has been done or paste transcripts from your team.",
            placeholder="E.g. Completed initial requirements gathering...",
        )
        if st.button("Add Update"):
            if update_text.strip():
                # Append update to session state
                st.session_state.updates += f"\n{update_text.strip()}"
                st.success("Update recorded.")
            else:
                st.warning("Please enter some text before adding the update.")

        st.divider()
        st.subheader("Chat with the Assistant")
        # Display conversation history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        # Chat input
        user_input = st.chat_input("Ask a question or request guidance…")
        if user_input:
            # Append user's message
            st.session_state.messages.append({"role": "user", "content": user_input})
            # Generate assistant response
            reply = _generate_chat_reply(
                messages=st.session_state.messages,
                scope=st.session_state.scope_text,
                schedule=st.session_state.schedule_text,
                updates=st.session_state.updates,
            )
            if reply:
                st.session_state.messages.append({"role": "assistant", "content": reply})
            # Regenerate progress summary
            st.session_state.progress = _generate_summary(
                scope=st.session_state.scope_text,
                schedule=st.session_state.schedule_text,
                updates=st.session_state.updates,
            )

    with col_right:
        st.subheader("Progress Summary")
        if st.session_state.progress:
            st.markdown(st.session_state.progress)
        else:
            st.info("Upload your documents and add some updates to generate a progress summary.")


if __name__ == "__main__":
    main()