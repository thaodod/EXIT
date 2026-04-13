# attn_comp

Minimal standalone AttnComp inference package.

This folder keeps only the pieces needed for compression inference:

- the packaged fine-tuned attention checkpoint
- the custom attention module used for scoring documents
- a small Python API for `question + chunks -> compressed chunks`
- a lightweight CLI and demo

## Python usage

```python
from attn_comp import AttnCompCompressor

compressor = AttnCompCompressor(
    model_name_or_path="/path/to/Meta-Llama-3.1-8B-Instruct",
)

result = compressor.compress(
    question="What are the advantages of Retrieval-Augmented Generation?",
    chunks=[
        "irrelevant document ...",
        "relevant document ...",
    ],
    p=0.9,
    epsilon=1e-2,
)

print(result.kept_indices)
print(result.compressed_context)
```

## CLI usage

Prepare a JSON file that contains a list of chunks:

```json
[
  "chunk one",
  "chunk two"
]
```

Run:

```bash
python -m attn_comp \
  --model-path /path/to/Meta-Llama-3.1-8B-Instruct \
  --question "What are the advantages of RAG?" \
  --chunks-file /path/to/chunks.json \
  --json
```

## Files

- `compressor.py`: public inference API
- `attention.py`: small custom attention module
- `checkpoints/llama-attention-layer13-SFT_epoch-7.pth`: bundled inference checkpoint
