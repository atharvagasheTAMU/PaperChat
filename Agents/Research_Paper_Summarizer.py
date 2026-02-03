import os
import re
import difflib
import json
import math
from operator import add as add_messages
from typing import Annotated, Sequence, TypedDict

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph

try:
    from Agents.llm_provider import get_embeddings, get_llm
except ImportError:
    from llm_provider import get_embeddings, get_llm

load_dotenv()

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PAPERS_DIR = os.getenv("PAPERS_DIR", os.path.join(BASE_DIR, "Papers"))
PERSIST_DIR = os.getenv("PAPERS_CHROMA_DIR", os.path.join(BASE_DIR, "chroma_papers"))
COLLECTION_NAME = os.getenv("PAPERS_COLLECTION", "research_papers")
VISUALS_DIR = os.getenv("PAPERS_VISUALS_DIR", os.path.join(BASE_DIR, "visuals"))
REBUILD_INDEX = os.getenv("REBUILD_PAPERS_INDEX", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}


def _require_pymupdf():
    try:
        import fitz  # type: ignore
    except Exception as exc:
        raise ImportError(
            "Visual extraction requires PyMuPDF. Install it with: pip install pymupdf"
        ) from exc
    return fitz


def _collect_pdf_paths(directory: str) -> list[str]:
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    pdfs = [
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if name.lower().endswith(".pdf")
    ]
    return pdfs


def _manifest_path() -> str:
    return os.path.join(PERSIST_DIR, "papers_manifest.json")


def _current_manifest(pdf_paths: list[str]) -> dict:
    items = []
    for path in pdf_paths:
        try:
            items.append(
                {
                    "name": os.path.basename(path),
                    "mtime": os.path.getmtime(path),
                    "size": os.path.getsize(path),
                }
            )
        except OSError:
            items.append({"name": os.path.basename(path), "mtime": 0, "size": 0})
    return {
        "index_version": 2,
        "loader": "pymupdf",
        "papers": sorted(items, key=lambda item: item["name"]),
    }


def _manifest_changed(pdf_paths: list[str]) -> bool:
    if not os.path.exists(PERSIST_DIR):
        return True
    manifest_file = _manifest_path()
    if not os.path.exists(manifest_file):
        return True
    try:
        with open(manifest_file, "r", encoding="utf-8") as handle:
            existing = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return True
    return existing != _current_manifest(pdf_paths)


def _write_manifest(pdf_paths: list[str]) -> None:
    os.makedirs(PERSIST_DIR, exist_ok=True)
    manifest_file = _manifest_path()
    with open(manifest_file, "w", encoding="utf-8") as handle:
        json.dump(_current_manifest(pdf_paths), handle, indent=2)


def _find_pdf_by_name(paper_name: str) -> str | None:
    pdf_paths = _collect_pdf_paths(PAPERS_DIR)
    needle = paper_name.strip().lower()
    for path in pdf_paths:
        if needle in os.path.basename(path).lower():
            return path
    return None


def get_paper_names() -> list[str]:
    return [os.path.basename(path) for path in _collect_pdf_paths(PAPERS_DIR)]


def _normalize_text(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    return " ".join(cleaned.lower().split())


def _extract_titles_from_pdf(pdf_path: str) -> dict:
    try:
        fitz = _require_pymupdf()
        doc = fitz.open(pdf_path)
        if not doc:
            return {"metadata": "", "page_text": ""}
        metadata_title = (doc.metadata or {}).get("title") or ""
        page = doc[0]
        lines = [line.strip() for line in page.get_text("text").splitlines() if line.strip()]
        if not lines:
            return {"metadata": metadata_title, "page_text": ""}
        top_lines = lines[:12]
        return {"metadata": metadata_title, "page_text": max(top_lines, key=len)}
    except Exception:
        return {"metadata": "", "page_text": ""}


def _build_paper_profiles() -> dict:
    profiles = {}
    embeddings = get_embeddings()
    for name in get_paper_names():
        path = _find_pdf_by_name(name)
        titles = _extract_titles_from_pdf(path) if path else {"metadata": "", "page_text": ""}
        profile_text = _extract_profile_text_from_pdf(path) if path else ""
        embedding = embeddings.embed_query(profile_text) if profile_text else None
        profiles[name] = {
            "titles": titles,
            "title_norms": {
                "metadata": _normalize_text(titles.get("metadata", "")),
                "page_text": _normalize_text(titles.get("page_text", "")),
            },
            "profile_text": profile_text,
            "profile_text_len": len(profile_text.strip()),
            "embedding": embedding,
        }
    return profiles


def _extract_profile_text_from_pdf(pdf_path: str) -> str:
    try:
        fitz = _require_pymupdf()
        doc = fitz.open(pdf_path)
        if not doc:
            return ""
        pages = [doc[i].get_text("text") for i in range(min(4, len(doc)))]
        text = "\n".join(pages)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(lines[:80])
    except Exception:
        return ""


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return -1.0
    dot = sum(x * y for x, y in zip(a, b))
    denom = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / denom if denom else -1.0


def _keyword_overlap_score(query: str, profile_text: str) -> float:
    query_tokens = {
        token
        for token in _normalize_text(query).split()
        if len(token) >= 4
    }
    if not query_tokens or not profile_text:
        return 0.0
    profile_tokens = set(_normalize_text(profile_text).split())
    if not profile_tokens:
        return 0.0
    overlap = len(query_tokens & profile_tokens)
    return overlap / max(len(query_tokens), 1)


def _visual_query_intent(query: str) -> bool:
    keywords = {"figure", "fig", "table", "image", "diagram", "visual", "plot", "chart", "example", "examples"}
    tokens = set(_normalize_text(query).split())
    return any(token in tokens for token in keywords)


def should_include_visuals(query: str, history: list[dict] | None = None) -> bool:
    """
    Use the LLM to decide if visuals should be included. Falls back to keyword intent.
    """
    try:
        llm = get_llm(temperature=0)
        history_text = ""
        if history:
            recent = [
                item.get("text", "")
                for item in history[-4:]
                if item.get("role") == "user"
            ]
            history_text = "\n".join(recent)

        system = (
            "You decide whether a user expects visual outputs (figures/tables/images). "
            "Answer ONLY 'yes' or 'no'."
        )
        prompt = (
            f"Conversation:\n{history_text}\n\n"
            f"User query:\n{query}\n\n"
            "Should the response include visuals?"
        )
        decision = llm.invoke(
            [SystemMessage(content=system), HumanMessage(content=prompt)]
        ).content.strip().lower()
        if decision.startswith("y"):
            return True
        return _visual_query_intent(query)
    except Exception:
        return _visual_query_intent(query)


def _visuals_relevant_to_query(query: str, records: list[dict]) -> bool:
    query_tokens = {
        token
        for token in _normalize_text(query).split()
        if len(token) >= 4
    }
    if not query_tokens:
        return False
    for record in records:
        captions = " ".join(record.get("captions", []))
        caption_tokens = set(_normalize_text(captions).split())
        if len(query_tokens & caption_tokens) >= 2:
            return True
    return False

def _infer_paper_from_query(query: str) -> str | None:
    paper_names = get_paper_names()
    if not paper_names:
        return None
    if len(paper_names) == 1:
        return paper_names[0]

    query_norm = _normalize_text(query)
    if not query_norm:
        return None

    title_scores = []
    for name, profile in PAPER_PROFILES.items():
        title_norms = profile.get("title_norms", {})
        for title_norm in title_norms.values():
            if not title_norm:
                continue
            ratio = difflib.SequenceMatcher(None, query_norm, title_norm).ratio()
            title_scores.append((ratio, name))

    if title_scores:
        title_scores.sort(reverse=True)
        if title_scores[0][0] >= 0.55:
            return title_scores[0][1]

    query_embedding = get_embeddings().embed_query(query)
    best_name = None
    best_score = -1.0
    best_keyword = 0.0
    for name, profile in PAPER_PROFILES.items():
        embedding = profile.get("embedding")
        profile_text = profile.get("profile_text", "")
        if not embedding:
            continue
        similarity = _cosine_similarity(query_embedding, embedding)
        keyword_score = _keyword_overlap_score(query, profile_text)
        combined = similarity + (0.15 * keyword_score)
        if combined > best_score:
            best_score = combined
            best_name = name
            best_keyword = keyword_score

    if best_name and best_score >= 0.18 and best_keyword >= 0.08:
        return best_name

    return None


def _retrieve_context(query: str, paper_name: str | None = None, k: int = 6):
    if paper_name:
        try:
            return vectorstore.similarity_search(
                query, k=k, filter={"source": paper_name}
            )
        except Exception:
            return []
    return vectorstore.similarity_search(query, k=k)


def _format_context(docs) -> str:
    lines = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "n/a")
        lines.append(f"[{i}] {source} (page {page})\n{doc.page_content}")
    return "\n\n".join(lines)


def _load_documents(pdf_paths: list[str]):
    documents = []
    for path in pdf_paths:
        loader = PyMuPDFLoader(path)
        pages = loader.load()
        for page in pages:
            page.metadata["source"] = os.path.basename(path)
        documents.extend(pages)
    return documents


def _build_or_load_vectorstore():
    embeddings = get_embeddings()
    pdf_paths = _collect_pdf_paths(PAPERS_DIR)

    if os.path.exists(PERSIST_DIR) and not REBUILD_INDEX:
        if not _manifest_changed(pdf_paths):
            return Chroma(
                persist_directory=PERSIST_DIR,
                embedding_function=embeddings,
                collection_name=COLLECTION_NAME,
            )

    if not pdf_paths:
        os.makedirs(PERSIST_DIR, exist_ok=True)
        _write_manifest(pdf_paths)
        return Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME,
        )

    documents = _load_documents(pdf_paths)
    if not documents:
        os.makedirs(PERSIST_DIR, exist_ok=True)
        _write_manifest(pdf_paths)
        return Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME,
        )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(documents)
    if not chunks:
        os.makedirs(PERSIST_DIR, exist_ok=True)
        _write_manifest(pdf_paths)
        return Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME,
        )
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
        collection_name=COLLECTION_NAME,
    )
    _write_manifest(pdf_paths)
    return vectorstore


vectorstore = _build_or_load_vectorstore()
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 6},
)
PAPER_PROFILES = _build_paper_profiles()

_VISUAL_PATTERN = re.compile(
    r"(?P<label>(Figure|Fig\.|Table))\s*\d+[:.\-]?\s*(?P<caption>.+)",
    re.IGNORECASE,
)


@tool
def list_papers() -> str:
    """List available research papers in the papers directory."""
    pdf_paths = _collect_pdf_paths(PAPERS_DIR)
    names = [os.path.basename(path) for path in pdf_paths]
    return "\n".join(names)


@tool
def search_papers(query: str) -> str:
    """Search papers and return relevant passages with citations."""
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant passages were found."

    results = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "n/a")
        results.append(
            f"[{i}] {source} (page {page})\n{doc.page_content}"
        )
    return "\n\n".join(results)


@tool
def find_visuals(query: str) -> str:
    """Find figure/table captions relevant to a query."""
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant passages were found."

    visuals = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "n/a")
        for line in doc.page_content.splitlines():
            match = _VISUAL_PATTERN.search(line.strip())
            if match:
                label = match.group("label")
                caption = match.group("caption").strip()
                visuals.append(
                    f"{label} — {caption} ({source}, page {page})"
                )

    if not visuals:
        return "No figure or table captions found in the retrieved passages."

    return "\n".join(dict.fromkeys(visuals))


def _extract_visuals_for_pdf(pdf_path: str, max_pages: int) -> str:
    records = _extract_visuals_records_for_pdf(pdf_path, max_pages)
    if not records:
        return "No figure/table captions found to extract."
    return "\n".join(_format_visual_records(records))


def _format_visual_records(records: list[dict]) -> list[str]:
    formatted = []
    for record in records:
        formatted.append(
            f"{record['path']} | page {record['page']} | "
            f"captions: {'; '.join(record['captions'])}"
        )
    return formatted


def _extract_visuals_records_for_pdf(
    pdf_path: str, max_pages: int
) -> list[dict]:
    fitz = _require_pymupdf()
    os.makedirs(VISUALS_DIR, exist_ok=True)
    doc = fitz.open(pdf_path)

    saved = []
    for page_index in range(len(doc)):
        page = doc[page_index]
        text = page.get_text("text")
        captions = []
        for line in text.splitlines():
            match = _VISUAL_PATTERN.search(line.strip())
            if match:
                label = match.group("label")
                caption = match.group("caption").strip()
                captions.append(f"{label} — {caption}")

        if not captions:
            continue

        images = page.get_images(full=True)
        if not images:
            continue

        for image_index, image in enumerate(images, start=1):
            xref = image[0]
            extracted = doc.extract_image(xref)
            image_bytes = extracted.get("image")
            image_ext = extracted.get("ext", "png")
            if not image_bytes:
                continue

            filename = (
                f"{os.path.splitext(os.path.basename(pdf_path))[0]}"
                f"_page_{page_index + 1}_img_{image_index}.{image_ext}"
            )
            output_path = os.path.join(VISUALS_DIR, filename)
            with open(output_path, "wb") as handle:
                handle.write(image_bytes)

            saved.append(
                {
                    "path": output_path,
                    "page": page_index + 1,
                    "captions": captions,
                }
            )
            if len(saved) >= max_pages:
                return saved

    return saved


def _extract_visuals_records_for_pdf_pages(
    pdf_path: str, page_numbers: list[int], max_images: int
) -> list[dict]:
    fitz = _require_pymupdf()
    os.makedirs(VISUALS_DIR, exist_ok=True)
    doc = fitz.open(pdf_path)

    saved = []
    for page_index in page_numbers:
        if page_index < 0 or page_index >= len(doc):
            continue
        page = doc[page_index]
        text = page.get_text("text")
        captions = []
        caption_lines = []
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for idx, line in enumerate(lines):
            match = _VISUAL_PATTERN.search(line)
            if match:
                label = match.group("label")
                caption = match.group("caption").strip()
                next_line = ""
                if idx + 1 < len(lines):
                    candidate = lines[idx + 1]
                    if not _VISUAL_PATTERN.search(candidate):
                        next_line = candidate
                # Keep captions to one line. Drop generic/empty ones.
                if len(caption.split()) < 4 or caption.strip() in {"-", "—", "."}:
                    continue
                caption_text = f"{label} — {caption}"
                captions.append(caption_text)
                caption_lines.append(line.strip())

        images = page.get_images(full=True)
        extracted_any = False
        for image_index, image in enumerate(images, start=1):
            xref = image[0]
            extracted = doc.extract_image(xref)
            image_bytes = extracted.get("image")
            image_ext = extracted.get("ext", "png")
            width = extracted.get("width", 0)
            height = extracted.get("height", 0)
            if not image_bytes or width < 50 or height < 50:
                continue

            # Skip images that are essentially blank/white
            try:
                pix = fitz.Pixmap(fitz.csRGB, fitz.open("png", image_bytes)[0].get_pixmap())
                samples = pix.samples
                if samples:
                    avg = sum(samples) / len(samples)
                    if avg > 250:
                        continue
            except Exception:
                pass

            filename = (
                f"{os.path.splitext(os.path.basename(pdf_path))[0]}"
                f"_page_{page_index + 1}_img_{image_index}.{image_ext}"
            )
            output_path = os.path.join(VISUALS_DIR, filename)
            with open(output_path, "wb") as handle:
                handle.write(image_bytes)

            saved.append(
                {
                    "path": output_path,
                    "page": page_index + 1,
                    "captions": captions,
                }
            )
            extracted_any = True
            if len(saved) >= max_images:
                return saved

        if not extracted_any and captions:
            clip_rect = None
            for caption_line in caption_lines:
                rects = page.search_for(caption_line)
                if rects:
                    # Use the first match and include a margin above it
                    rect = rects[0]
                    top = max(rect.y0 - 420, 0)
                    bottom = min(rect.y1 + 40, page.rect.height)
                    clip_rect = fitz.Rect(0, top, page.rect.width, bottom)
                    break
            if clip_rect is None or (clip_rect.height / page.rect.height) > 0.6:
                continue
            # Skip dense text blocks; keep only visually light figure areas.
            try:
                text_rects = page.search_for(" ", quads=False)
                text_density = len(text_rects) / max(clip_rect.height * clip_rect.width, 1)
                if text_density > 0.0008:
                    continue
            except Exception:
                pass
            pix = page.get_pixmap(dpi=200, clip=clip_rect)
            if pix.samples:
                avg = sum(pix.samples) / len(pix.samples)
                if avg > 250:
                    continue
            filename = (
                f"{os.path.splitext(os.path.basename(pdf_path))[0]}"
                f"_page_{page_index + 1}_render.png"
            )
            output_path = os.path.join(VISUALS_DIR, filename)
            pix.save(output_path)
            saved.append(
                {
                    "path": output_path,
                    "page": page_index + 1,
                    "captions": captions,
                }
            )
            if len(saved) >= max_images:
                return saved

    return saved


@tool
def extract_visuals(
    paper_name: str | None = None,
    max_pages: int = 3,
    output_path: str | None = None,
) -> str:
    """
    Extract embedded images from a specific paper by name.
    If the paper name is not provided, use extract_visuals_for_query.
    """
    if not paper_name:
        return (
            "Please provide paper_name, e.g. "
            "extract_visuals(paper_name='Spatial transformer Network'). "
            "Or use extract_visuals_for_query(query='STN network')."
        )

    pdf_path = _find_pdf_by_name(paper_name)
    if not pdf_path:
        return f"No matching paper found for: {paper_name}"

    return _extract_visuals_for_pdf(pdf_path, max_pages)


@tool
def extract_visuals_for_query(query: str, max_pages: int = 3) -> str:
    """
    Extract visuals by inferring the most relevant paper from a query.
    """
    best_source = _infer_paper_from_query(query)
    if not best_source:
        return "No relevant passages found to identify a paper."
    pdf_path = _find_pdf_by_name(best_source)
    if not pdf_path:
        return f"No matching paper found for: {best_source}"
    docs = _retrieve_context(query, paper_name=best_source, k=6)
    page_numbers = []
    for doc in docs:
        page = doc.metadata.get("page")
        if isinstance(page, int):
            page_numbers.append(page)
        elif isinstance(page, str) and page.isdigit():
            page_numbers.append(int(page))
    page_numbers = list(dict.fromkeys(page_numbers))
    if page_numbers:
        records = _extract_visuals_records_for_pdf_pages(
            pdf_path, page_numbers, max_images=max_pages
        )
        if records:
            return "\n".join(_format_visual_records(records))
    return _extract_visuals_for_pdf(pdf_path, max_pages)


def extract_visuals_records_for_query(
    query: str,
    max_pages: int = 3,
    history: list[dict] | None = None,
) -> list[dict]:
    composed = _compose_query(query, history)
    best_source = _infer_paper_from_query(composed)
    if not best_source:
        return []
    pdf_path = _find_pdf_by_name(best_source)
    if not pdf_path:
        return []
    docs = _retrieve_context(query, paper_name=best_source, k=6)
    page_numbers = []
    for doc in docs:
        page = doc.metadata.get("page")
        if isinstance(page, int):
            page_numbers.append(page)
        elif isinstance(page, str) and page.isdigit():
            page_numbers.append(int(page))
    page_numbers = list(dict.fromkeys(page_numbers))
    if page_numbers:
        records = _extract_visuals_records_for_pdf_pages(
            pdf_path, page_numbers, max_images=max_pages
        )
        if records and (_visual_query_intent(query) or _visuals_relevant_to_query(query, records)):
            return records
        return []
    records = _extract_visuals_records_for_pdf(pdf_path, max_pages)
    if records and (
        _visual_query_intent(query) or _visuals_relevant_to_query(query, records)
    ):
        return records
    return []


def extract_visuals_records_for_paper(
    paper_name: str, max_pages: int = 3
) -> list[dict]:
    pdf_path = _find_pdf_by_name(paper_name)
    if not pdf_path:
        return []
    return _extract_visuals_records_for_pdf(pdf_path, max_pages)


tools = [list_papers, search_papers, find_visuals, extract_visuals, extract_visuals_for_query]
llm = get_llm(temperature=0, tools=tools)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def _should_continue(state: AgentState):
    last = state["messages"][-1]
    return hasattr(last, "tool_calls") and len(last.tool_calls) > 0


system_prompt = """
You are a research paper summarizer chatbot.
Use the available tools to retrieve relevant passages before answering.
When summarizing a paper, return this structure:
- Title (if known)
- Objective
- Methods
- Results
- Limitations
- Key Takeaways
Include citations inline using the format [n] from tool results.
If visuals are requested, call extract_visuals with a paper_name. If the user
does not specify a paper, call extract_visuals_for_query.
Include the output paths in your response. You can call find_visuals to list captions.
If the user asks "what papers are available", call list_papers.
"""

tools_dict = {t.name: t for t in tools}


def call_llm(state: AgentState) -> AgentState:
    messages = [SystemMessage(content=system_prompt)] + list(state["messages"])
    response = llm.invoke(messages)
    return {"messages": [response]}


def _compose_query(message: str, history: list[dict] | None) -> str:
    if not history:
        return message
    recent = []
    for item in history[-4:]:
        if item.get("role") == "user" and item.get("text"):
            recent.append(item["text"])
    return "\n".join(recent + [message]) if recent else message


def answer_question(message: str, history: list[dict] | None = None) -> str:
    composed = _compose_query(message, history)
    inferred = _infer_paper_from_query(composed)
    docs = _retrieve_context(message, paper_name=inferred)
    if not docs:
        if inferred:
            profile = PAPER_PROFILES.get(inferred, {})
            if profile.get("profile_text_len", 0) == 0:
                return (
                    f"I matched this request to '{inferred}', but could not extract any "
                    "readable text from that PDF. It may be scanned or image-only. "
                    "If you want, I can enable OCR to extract the text."
                )
        return (
            "I couldn't find this in the indexed papers. "
            "If you want, add the relevant PDF or rephrase the question."
        )

    context = _format_context(docs)
    system = (
        "You are a research paper summarizer. Answer using ONLY the context "
        "below. If the answer is not in the context, say you could not find it. "
        "Cite sources inline using numeric brackets like [1], [2]. "
        "Never use [n]. If you cannot cite, omit citations. "
        "Avoid meta phrases like \"based on the context\" or \"the paper says\". "
        "Be direct and natural."
    )
    prompt = f"{system}\n\nContext:\n{context}\n\nQuestion:\n{message}"
    llm = get_llm(temperature=0)
    return llm.invoke([SystemMessage(content=system), HumanMessage(content=prompt)]).content


def reload_indexes() -> None:
    global vectorstore, retriever, PAPER_PROFILES
    vectorstore = _build_or_load_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6},
    )
    PAPER_PROFILES = _build_paper_profiles()


def take_action(state: AgentState) -> AgentState:
    tool_calls = state["messages"][-1].tool_calls
    results = []
    for call in tool_calls:
        tool_name = call["name"]
        if tool_name not in tools_dict:
            result = "Unknown tool name. Please retry."
        else:
            result = tools_dict[tool_name].invoke(call["args"])
        results.append(
            ToolMessage(
                tool_call_id=call["id"],
                name=tool_name,
                content=str(result),
            )
        )
    return {"messages": results}


graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("tools", take_action)
graph.add_conditional_edges("llm", _should_continue, {True: "tools", False: END})
graph.add_edge("tools", "llm")
graph.set_entry_point("llm")

app = graph.compile()


def run_chatbot():
    print("\n=== Research Paper Summarizer ===")
    print(f"Papers folder: {PAPERS_DIR}")
    print("Type 'exit' to quit.")

    while True:
        user_input = input("\nAsk a question or request a summary: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break

        visual_keywords = ("visual", "figure", "fig", "table", "image", "diagram")
        if any(word in user_input.lower() for word in visual_keywords):
            print("\n=== ANSWER ===")
            print(extract_visuals_for_query.invoke({"query": user_input}))
            continue

        messages = [HumanMessage(content=user_input)]
        result = app.invoke({"messages": messages})
        print("\n=== ANSWER ===")
        print(result["messages"][-1].content)


if __name__ == "__main__":
    run_chatbot()

