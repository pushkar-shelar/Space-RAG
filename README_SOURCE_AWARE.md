# Space RAG — Corpus + Private Session Architecture

## Intended behavior

Space RAG now has two evidence stores:

1. **Permanent corpus**: PDFs already present in `data/corpus/`.
2. **Private session document**: the PDF uploaded by the current user/session.

A private session can retrieve from both stores.

### User-facing source policy

- Evidence from the **permanent corpus** can be used to answer the question, but the answer/UI must not expose corpus filenames, paths, page numbers, or internal source identifiers.
- Evidence from the **private uploaded PDF** must expose its document name and page number so the user can see where the information came from.

### Privacy

A private session is stored under:

```text
data/runtime_sessions/<session_id>/
```

When the session ends, the whole directory is deleted.

The permanent corpus is never deleted by private-session cleanup.

## New data layout

```text
data/
├── corpus/                         # permanent corpus PDFs
├── corpus_index/                   # permanent derived corpus indexes
│   ├── documents/
│   ├── chroma/
│   ├── bm25.pkl
│   ├── colpali.pt
│   └── corpus_manifest.json
└── runtime_sessions/               # temporary private sessions
    └── <session_id>/
        ├── document/
        ├── text/
        ├── pages/
        ├── images/
        ├── metadata/
        ├── chroma/
        └── indexes/
            ├── bm25.pkl
            └── colpali.pt
```

The old shared `data/processed` retrieval data is no longer used by this architecture.

## One-time corpus build

Put the permanent PDFs in:

```text
data/corpus/
```

Then run:

```powershell
python -m src.indexing.corpus_builder
```

This builds the permanent BM25, Chroma, and ColPali indexes.

The Streamlit UI also has a **Build / Refresh Corpus Index** button.

## Private session

Upload a PDF in Streamlit.

The upload is copied into the new session directory, not into `data/corpus`.

Text retrieval indexes are built immediately. ColPali for the private document is built lazily when the first visual/multimodal query arrives, to reduce startup time.

## LLM modes

### Offline
Uses the local Qwen model:

```text
Qwen/Qwen2.5-1.5B-Instruct
```

### Online
Uses Grok through the xAI API.

Set:

```powershell
$env:XAI_API_KEY="YOUR_KEY"
```

Install the OpenAI-compatible client:

```powershell
pip install -U openai
```

The application sends retrieved evidence only. It does not upload the raw private PDF to Grok.

## Source-aware answer behavior

Internally every result has:

```text
source_type = "corpus"
```

or

```text
source_type = "private"
```

For private results, citations are generated as:

```text
[document.pdf, p. 7]
```

Corpus provenance is deliberately removed from the LLM-facing context except for the fact that it is permanent corpus evidence. Therefore the model can use the information without exposing where in the corpus it came from.

## Existing verification module

Keep the existing:

```text
src/verification/cross_modal_verifier.py
```

The new multimodal retriever calls it only for visual/multimodal queries.

## Run

From the project root:

```powershell
streamlit run app/streamlit_app.py
```

## Important migration note

The old directories can remain on disk while migrating, but the new application does not query the old `data/processed` indexes. After the new architecture is verified, old derived data can be removed manually if it is no longer needed.
