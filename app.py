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

# app.py
import streamlit as st, pandas as pd, json
from datetime import datetime
from sqlalchemy import create_engine, text
from openai import OpenAI

st.set_page_config(page_title="AI-safe List Editor", layout="wide")
client = OpenAI()  # set OPENAI_API_KEY env var

# ---- DB setup (SQLite now; Postgres later) ----
# For Postgres later: "postgresql+psycopg://user:pass@host:5432/dbname"
engine = create_engine("sqlite:///app.db", future=True, echo=False)

def init_db():
    with engine.begin() as con:
        con.exec_driver_sql("""
        CREATE TABLE IF NOT EXISTS docs(
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          name TEXT, type TEXT, source_path TEXT, created_at TEXT
        );
        """)
        con.exec_driver_sql("""
        CREATE TABLE IF NOT EXISTS items(
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          doc_id INTEGER, item_text TEXT NOT NULL,
          status TEXT CHECK(status IN ('todo','in_progress','done','dropped')) DEFAULT 'todo',
          note TEXT DEFAULT '', last_updated TEXT
        );
        """)
        con.exec_driver_sql("""
        CREATE TABLE IF NOT EXISTS actions_log(
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          ts TEXT, user_text TEXT, model TEXT, actions_json TEXT
        );
        """)
        # optional now; used when you add context docs
        con.exec_driver_sql("""
        CREATE TABLE IF NOT EXISTS chunks(
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          doc_id INTEGER, chunk_text TEXT, embedding BLOB
        );
        """)
init_db()

@st.cache_data
def load_items(doc_id:int):
    with engine.begin() as con:
        df = pd.read_sql(text("SELECT id, item_text, status, note, last_updated FROM items WHERE doc_id=:d"), con, params={"d": doc_id})
    return df

def save_actions_log(user_text, model, actions):
    with engine.begin() as con:
        con.execute(text("INSERT INTO actions_log(ts,user_text,model,actions_json) VALUES(:ts,:ut,:m,:aj)"),
                    {"ts": datetime.utcnow().isoformat(), "ut": user_text, "m": model, "aj": json.dumps(actions)})

def apply_actions(doc_id:int, actions:list):
    with engine.begin() as con:
        for a in actions:
            op = a.get("op"); iid = a.get("item_id")
            if not isinstance(iid, int): continue
            if op in ("mark_done","mark_dropped","set_status","add_note"):
                # status changes (never touch item_text)
                if op == "mark_done":
                    con.execute(text("UPDATE items SET status='done', last_updated=:ts WHERE id=:id AND doc_id=:d"),
                                {"ts": datetime.utcnow().isoformat(), "id": iid, "d": doc_id})
                elif op == "mark_dropped":
                    con.execute(text("UPDATE items SET status='dropped', last_updated=:ts WHERE id=:id AND doc_id=:d"),
                                {"ts": datetime.utcnow().isoformat(), "id": iid, "d": doc_id})
                elif op == "set_status":
                    s = a.get("status")
                    if s in ("todo","in_progress","done","dropped"):
                        con.execute(text("UPDATE items SET status=:s, last_updated=:ts WHERE id=:id AND doc_id=:d"),
                                    {"s": s, "ts": datetime.utcnow().isoformat(), "id": iid, "d": doc_id})
                # optional note
                if a.get("note"):
                    con.execute(text("UPDATE items SET note=:n, last_updated=:ts WHERE id=:id AND doc_id=:d"),
                                {"n": a["note"], "ts": datetime.utcnow().isoformat(), "id": iid, "d": doc_id})

# ---- UI: choose or create a doc ----
st.sidebar.header("Document")
doc_name = st.sidebar.text_input("Doc name", value="My List")
if st.sidebar.button("Create/Load doc"):
    with engine.begin() as con:
        # upsert-ish
        row = con.execute(text("SELECT id FROM docs WHERE name=:n"), {"n": doc_name}).fetchone()
        if row: st.session_state.doc_id = row[0]
        else:
            con.execute(text("INSERT INTO docs(name,type,source_path,created_at) VALUES(:n,'list','',:ts)"),
                        {"n": doc_name, "ts": datetime.utcnow().isoformat()})
            st.session_state.doc_id = con.execute(text("SELECT last_insert_rowid()")).scalar()

if "doc_id" not in st.session_state:
    st.info("Create/Load a doc in the sidebar to begin.")
    st.stop()

doc_id = st.session_state.doc_id
st.title(f"AI-safe List Editor — {doc_name}")

# Quick add new row (immutable text)
with st.expander("Add item"):
    new_text = st.text_input("Item text")
    if st.button("Add"):
        if new_text.strip():
            with engine.begin() as con:
                con.execute(text("""
                INSERT INTO items(doc_id,item_text,status,note,last_updated)
                VALUES(:d,:t,'todo','',:ts)
                """), {"d": doc_id, "t": new_text.strip(), "ts": datetime.utcnow().isoformat()})
            st.success("Added.")
            st.experimental_rerun()

# Show grid
df = load_items(doc_id)
st.dataframe(df, hide_index=True, use_container_width=True)

# ---- AI: Edits (JSON-only) vs Explain (read-only) ----
model = st.selectbox("Model", ["gpt-4.1-mini","o4-mini"], index=0)
colA, colB = st.columns(2)

def edits_prompt(rows, instruction):
    return f"""
SYSTEM: You must propose SAFE, STRUCTURED updates to a task list. Never output prose.
Rules:
- Do NOT modify 'item_text'.
- Only change 'status' or 'note'.
- If instruction is ambiguous, propose no actions.
Return ONLY JSON with:
{{"actions":[{{"op":"mark_done"|"mark_dropped"|"add_note"|"set_status","item_id":<int>,"note":<string?>,"status":<"todo"|"in_progress"|"done"|"dropped"?>}}]}}
If nothing to change, return {{"actions":[]}}.

USER DATA: {json.dumps(rows, ensure_ascii=False)}
USER INSTRUCTION: {instruction}
"""

def explain_prompt(rows, question):
    return f"""
SYSTEM: You are a summarizer. Answer the question about the list. Output prose ONLY (no JSON).
DATA: {json.dumps(rows, ensure_ascii=False)}
QUESTION: {question}
"""

with colA:
    st.subheader("Edits mode")
    instruction = st.text_area("Tell the model what to do (safe edits):", placeholder="Mark 'Site visit' as dropped and add a note explaining it's out of scope.")
    if st.button("Preview AI Edits"):
        rows = df.to_dict(orient="records")
        resp = client.responses.create(model=model, input=edits_prompt(rows, instruction))
        raw = resp.output_text.strip()
        try:
            parsed = json.loads(raw)
            actions = parsed.get("actions", [])
        except Exception:
            actions = []
        st.session_state["preview_actions"] = actions

    if "preview_actions" in st.session_state and st.session_state["preview_actions"]:
        st.info("Preview (nothing applied yet):")
        st.code(json.dumps(st.session_state["preview_actions"], indent=2))
        c1, c2 = st.columns(2)
        if c1.button("Apply actions"):
            save_actions_log(instruction, model, st.session_state["preview_actions"])
            apply_actions(doc_id, st.session_state["preview_actions"])
            st.session_state["preview_actions"] = []
            st.experimental_rerun()
        if c2.button("Discard"):
            st.session_state["preview_actions"] = []
            st.experimental_rerun()

with colB:
    st.subheader("Explain mode (read-only)")
    q = st.text_input("Ask about the list:", placeholder="What’s left on the list?")
    if st.button("Explain"):
        rows = df.to_dict(orient="records")
        resp = client.responses.create(model=model, input=explain_prompt(rows, q))
        st.write(resp.output_text)
