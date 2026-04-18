```
Data Alignment
The script should not rely on qa_only.json order alone unless it matches records.jsonl. Safer plan:...
```
> I think qa_only.json is the original dataset file. The chunk db building also come from the this original file. also DB doesn't have question I think, it only store the embedding forms of chunks, Right? 

```This avoids silent mismatch if chunk DB was built with --limit or a different subset.```

> actually the "--limit" is for smoke test. so don't worry about this. 

So, about the `Data Alignment` section should be reconsider if needed.

at the section `Retrieval`, you mentioned this `also produce order-preserving merged raw context` ? What does that mean ? I thought we no longer doing merging.

at the section `Prompt Generation`:
I think we can use new prompt templates for both cases. No need to peg with old prompts because I will use thinking models as LLM reader (such as Qwen3.5-Flash).

Confirmation Points:
1) For compressor input, should we use retrieval-rank order? YES, OF COURSE, what other rank in this story ???

2) Should method=none be included as a raw retrieved-context baseline? YES, it is made for raw (not raw original context, it is uncompressed but following retrieval order)