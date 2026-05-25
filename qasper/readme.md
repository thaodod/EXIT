# QASPER Retrieved Benchmark

This directory contains the preprocessing/retrieval script used to create the
QASPER benchmark files in `retrieved/contriever-msmarco_QASPER/`.

## Source

- Dataset: QASPER v0.3, official AllenAI archives.
- Train/dev archive: `https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz`
- Test archive: `https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz`
- Raw archives are cached locally under `qasper/raw/`.

## Output

Generated files:

- `retrieved/contriever-msmarco_QASPER/dev.json`
- `retrieved/contriever-msmarco_QASPER/dev_500.json`
- `retrieved/contriever-msmarco_QASPER/test.json`
- `retrieved/contriever-msmarco_QASPER/test_500.json`
- `retrieved/contriever-msmarco_QASPER/metadata.json`

The `.json` data files follow the repository's existing JSONL convention: one
JSON object per line, with `question`, `answers`, `answer`, and `ctxs`.

The generated dev/test files still contain exactly 100 retrieved `ctxs` per
question. In the retrieved top-100 contexts, word counts are centered around the
target: dev median 101 / mean 100.51 / max 281, and test median 101 / mean
100.52 / max 250.

## Settings

- Chunk source: all QASPER train, dev, and test papers are used as the retrieval
  corpus. The corpus has 1,585 papers and 86,755 chunks.
- Text used for chunking: paper title, abstract, section names, and full-text
  paragraphs.
- Chunking: sentence-aware soft windows.
- Target chunk size: about 100 whitespace words.
- Overlap: whole trailing sentences totaling at least 15 words when possible,
  matching 15% of the 100-word target.
- Corpus chunk word stats: min 16, p05 85, mean 100.19, median 100, p95 115,
  max 1,157. The maximum is one appendix/list chunk that is kept whole because
  it has no reliable sentence break.
- Retriever: `facebook/contriever-msmarco`.
- Embedding: Hugging Face `AutoModel`, mean pooling over the last hidden state,
  dot-product scoring, no L2 normalization.
- Top-k: 100 chunks per question.
- Retrieval query: paper title plus original question (`title_question`). QASPER
  is paper-anchored, so the title is included to preserve that setting while
  still retrieving from the global QASPER scientific-paper corpus.
- Saved `question`: the original QASPER question only.

## Answer Filtering

QASPER has multiple answer annotations per question. I used the default
`--unanswerable-policy all`:

- Drop a question only when all answer annotations are marked
  `unanswerable: true`.
- For mixed questions, keep the question and remove only the unanswerable
  annotations.
- Build `answers` from remaining extractive spans, yes/no answers, and free-form
  answers, deduplicated while preserving order.

This produced:

- Dev: 1,005 raw questions, 945 kept, 60 dropped.
- Test: 1,451 raw questions, 1,372 kept, 79 dropped.

For a stricter variant that drops any question with any unanswerable annotation,
run with `--unanswerable-policy any`.

## `ctxs` Format

Each retrieved context has the same fields as the existing benchmark files:

```json
{
  "id": "qasper:<paper_id>:<chunk_index>",
  "title": "<paper title> | <dominant section>",
  "text": "<sentence-aware passage around 100 words>",
  "score": "<contriever dot-product score>",
  "hasanswer": true
}
```

`hasanswer` is true when the chunk contains a non-yes/no answer string or has
high token overlap with QASPER gold evidence/highlighted evidence.

## Reproduce

From the repository root:

```bash
conda run -n 312 python qasper/build_retrieved_qasper.py --overwrite
```

The script caches chunk embeddings under `qasper/cache/`, so later reruns reuse
the index unless `--rebuild-index` is passed.
