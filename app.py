import json
import logging
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

logging.basicConfig(level=logging.INFO)

st.set_page_config(
    page_title="Forensic Investigation Platform",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

def _new_session_id() -> str:
    return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

if "session_id" not in st.session_state:
    st.session_state.session_id = _new_session_id()
if "searcher" not in st.session_state:
    from semantic_search import SemanticForensicSearcher
    st.session_state.searcher = SemanticForensicSearcher()
if "feedback_manager" not in st.session_state:
    from feedback_manager import FeedbackManager
    st.session_state.feedback_manager = FeedbackManager()
if "current_results" not in st.session_state:
    st.session_state.current_results = []
if "last_query" not in st.session_state:
    st.session_state.last_query = ""

st.title("🔍 Forensic Investigation Platform")
st.markdown("*AI-Powered Keyword Search · Semantic Understanding · Active Learning*")

with st.sidebar:
    st.header("⚙️ Configuration")
    st.subheader("Search Settings")
    use_synonyms = st.checkbox("🧠 Synonym Expansion (WordNet)", value=True)
    use_feedback = st.checkbox("📊 Personalisation (SVM Feedback)", value=True)
    top_k = st.slider("Results to return", min_value=5, max_value=50, value=10)

    st.divider()
    st.subheader("📈 Session Statistics")
    stats = st.session_state.feedback_manager.get_session_stats(st.session_state.session_id)
    c1, c2, c3 = st.columns(3)
    c1.metric("Total", stats["total_feedback"])
    c2.metric("👍", stats["positive"])
    c3.metric("👎", stats["negative"])
    rate = stats["positive_rate"]
    if rate > 0:
        st.progress(rate, text=f"Relevance rate: {rate*100:.0f}%")
    else:
        st.caption("No feedback recorded yet.")

    st.divider()
    st.caption(f"Session: `{st.session_state.session_id}`")
    if st.button("🔄 New Session", type="secondary"):
        st.session_state.session_id = _new_session_id()
        st.session_state.current_results = []
        st.session_state.last_query = ""
        st.rerun()

tab_search, tab_upload, tab_folder, tab_report = st.tabs(
    ["🔎 Search", "📎 Upload Documents", "📁 Folder Ingest", "📊 Report"]
)

with tab_search:
    col_q, col_btn = st.columns([5, 1])
    with col_q:
        search_query = st.text_input(
            "Search query",
            placeholder="e.g. suspicious financial transfer | confidential agreement",
            label_visibility="collapsed",
        )
    with col_btn:
        do_search = st.button("🔍 Search", type="primary", use_container_width=True)

    if do_search:
        if not search_query.strip():
            st.warning("Please enter a search query.")
        elif st.session_state.searcher.index.ntotal == 0:
            st.warning("⚠️ No documents indexed yet. Use the Upload or Folder Ingest tabs first.")
        else:
            with st.spinner("Running semantic search…"):
                results = st.session_state.searcher.search(
                    query=search_query,
                    session_id=st.session_state.session_id,
                    top_k=top_k,
                    use_synonyms=use_synonyms,
                    use_feedback=use_feedback,
                )
            st.session_state.current_results = results
            st.session_state.last_query = search_query

    if st.session_state.current_results:
        st.subheader(f"Results for: *{st.session_state.last_query}*")
        for result in st.session_state.current_results:
            with st.container(border=True):
                left, mid, right = st.columns([4, 1, 1])
                with left:
                    st.markdown(f"**#{result['rank']}** &nbsp; 📄 `{result['document_name']}`")
                    if result.get("source"):
                        st.caption(f"Source: {result['source']}")
                with mid:
                    score = result["similarity_score"]
                    colour = "🟢" if score >= 0.7 else "🟡" if score >= 0.4 else "🔴"
                    st.markdown(f"{colour} **{score:.3f}**")
                with right:
                    ps = result.get("personalized_score")
                    if ps is not None:
                        st.caption(f"Personalised: {ps:.3f}")

                st.markdown(f"> {result['preview']}")

                fb1, fb2, _ = st.columns([1, 1, 5])
                with fb1:
                    if st.button("👍 Relevant", key=f"pos_{result['document_id']}_{result['rank']}"):
                        st.session_state.feedback_manager.add_feedback(
                            st.session_state.session_id,
                            st.session_state.last_query,
                            result["document_id"],
                            result["preview"],
                            is_relevant=True,
                        )
                        st.success("Feedback recorded — rankings will improve.")
                        st.rerun()
                with fb2:
                    if st.button("👎 Not Relevant", key=f"neg_{result['document_id']}_{result['rank']}"):
                        st.session_state.feedback_manager.add_feedback(
                            st.session_state.session_id,
                            st.session_state.last_query,
                            result["document_id"],
                            result["preview"],
                            is_relevant=False,
                        )
                        st.success("Feedback recorded — rankings will improve.")
                        st.rerun()

with tab_upload:
    st.subheader("📂 Upload and Index Documents")
    st.markdown("Supported formats: **TXT · PDF · DOCX · HTML**.")

    uploaded_files = st.file_uploader(
        "Choose files",
        type=["txt", "pdf", "docx", "html"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files and st.button("⚙️ Index Documents", type="primary"):
        from text_extractor import TextExtractor

        os.makedirs("data/temp", exist_ok=True)
        documents = []
        progress = st.progress(0, text="Extracting text…")

        for i, f in enumerate(uploaded_files):
            temp_path = os.path.join("data/temp", f.name)
            with open(temp_path, "wb") as fh:
                fh.write(f.getbuffer())

            text = TextExtractor.extract(temp_path)
            if text.strip():
                documents.append({"file_path": temp_path, "text": text, "source": "upload"})
                st.success(f"✅ {f.name} ({len(text):,} chars)")
            else:
                st.warning(f"⚠️ No text extracted from {f.name}")

            progress.progress((i + 1) / len(uploaded_files), text=f"Processing {f.name}…")

        if documents:
            with st.spinner("Encoding and indexing…"):
                st.session_state.searcher.add_documents(documents)
            st.success(f"✅ Indexed {len(documents)} document(s). Total in index: {st.session_state.searcher.index.ntotal}")
        progress.empty()

    st.divider()
    st.subheader("📚 Indexed Documents")
    docs = st.session_state.searcher.documents
    if docs:
        for doc in docs[-15:]:
            st.caption(f"• `{doc.get('file_path', 'unknown')}` &nbsp;[{doc.get('source', '?')}]")
        if len(docs) > 15:
            st.caption(f"…and {len(docs) - 15} more")
    else:
        st.info("No documents indexed yet.")

with tab_folder:
    st.subheader("📁 Folder Ingest")
    st.markdown("Scan a folder (recursively) for supported files and index them.")

    folder_path = st.text_input(
        "Folder path",
        placeholder=r"C:\Users\You\Documents\Evidence   or   /home/you/evidence",
    )

    recursive = st.checkbox("🔁 Scan sub-folders recursively", value=True)
    show_skipped = st.checkbox("📋 Show skipped files", value=False)

    SUPPORTED_EXTS = {".txt", ".pdf", ".docx", ".html", ".htm"}

    if st.button("📂 Scan and Index Folder", type="primary"):
        if not folder_path.strip():
            st.error("Please enter a folder path.")
        elif not os.path.isdir(folder_path):
            st.error(f"Folder not found: `{folder_path}`")
        else:
            from text_extractor import TextExtractor

            found_files = []
            if recursive:
                for root, _, files in os.walk(folder_path):
                    for fname in files:
                        if os.path.splitext(fname)[1].lower() in SUPPORTED_EXTS:
                            found_files.append(os.path.join(root, fname))
            else:
                for fname in os.listdir(folder_path):
                    full = os.path.join(folder_path, fname)
                    if os.path.isfile(full) and os.path.splitext(fname)[1].lower() in SUPPORTED_EXTS:
                        found_files.append(full)

            if not found_files:
                st.warning(f"No supported files found in `{folder_path}`.")
            else:
                st.info(f"Found **{len(found_files)}** file(s). Extracting text…")
                progress_bar = st.progress(0, text="Starting…")
                documents = []
                skipped = []

                for i, fpath in enumerate(found_files):
                    progress_bar.progress(
                        (i + 1) / len(found_files),
                        text=f"Reading {os.path.basename(fpath)} ({i+1}/{len(found_files)})",
                    )
                    text = TextExtractor.extract(fpath)
                    if text.strip():
                        documents.append({"file_path": fpath, "text": text, "source": "folder"})
                    else:
                        skipped.append(fpath)

                progress_bar.empty()

                if documents:
                    with st.spinner(f"Encoding and indexing {len(documents)} document(s)…"):
                        st.session_state.searcher.add_documents(documents)
                    st.success(f"✅ Indexed **{len(documents)}** document(s). Total in index: **{st.session_state.searcher.index.ntotal}**")
                else:
                    st.warning("No documents could be extracted from the folder.")

                if skipped and show_skipped:
                    with st.expander(f"⚠️ {len(skipped)} file(s) skipped (no text extracted)"):
                        for s in skipped:
                            st.caption(f"• {s}")

    st.divider()
    st.subheader("📚 Folder-Indexed Documents")
    folder_docs = [d for d in st.session_state.searcher.documents if d.get("source") == "folder"]
    if folder_docs:
        for doc in folder_docs[-20:]:
            st.caption(f"• `{doc['file_path']}`")
        if len(folder_docs) > 20:
            st.caption(f"…and {len(folder_docs) - 20} more")
    else:
        st.info("No folder documents indexed yet.")

with tab_report:
    st.subheader("📊 Forensic Investigation Report")
    if not st.session_state.current_results:
        st.info("Run a search first to populate the report.")
    else:
        if st.button("📄 Generate Report", type="primary"):
            from report_generator import ReportGenerator

            paths = ReportGenerator.generate_report(
                results=st.session_state.current_results,
                session_stats=st.session_state.feedback_manager.get_session_stats(st.session_state.session_id),
                session_id=st.session_state.session_id,
                last_query=st.session_state.last_query,
                indexed_doc_count=len(st.session_state.searcher.documents),
            )

            st.success("Report generated!")

            with open(paths["html_path"], "r", encoding="utf-8") as fh:
                html_data = fh.read()
            with open(paths["json_path"], "r", encoding="utf-8") as fh:
                json_data = fh.read()

            dl1, dl2 = st.columns(2)
            with dl1:
                st.download_button(
                    "⬇️ Download HTML Report",
                    data=html_data,
                    file_name=os.path.basename(paths["html_path"]),
                    mime="text/html",
                    type="primary",
                )
            with dl2:
                st.download_button(
                    "⬇️ Download JSON Report",
                    data=json_data,
                    file_name=os.path.basename(paths["json_path"]),
                    mime="application/json",
                )

            with st.expander("Preview HTML report"):
                st.components.v1.html(html_data, height=600, scrolling=True)

st.divider()
st.caption("Forensic Investigation Platform · Semantic Search (all-MiniLM-L6-v2 + FAISS) · Active Learning (Linear SVM) · Query Expansion (WordNet)")