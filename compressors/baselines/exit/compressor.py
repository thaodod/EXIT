"""EXIT implementation for context-aware extractive compression."""

import torch
import spacy
from typing import List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from functools import lru_cache
from ...base import BaseCompressor, SearchResult

class EXITCompressor(BaseCompressor):
    """EXIT: Context-aware extractive compression."""
    
    def __init__(
        self,
        base_model: str = "google/gemma-2b-it",
        checkpoint="doubleyyh/exit-gemma-2b",
        device='cuda',
        cache_dir: str = "./cache",
        batch_size: int = 8,
        threshold: float = 0.5
    ):
        """Initialize EXIT compressor.
        
        Args:
            base_model: Base model path
            checkpoint: Path to trained checkpoint
            device: Device to use (None for auto)
            cache_dir: Cache directory for models
            batch_size: Batch size for processing
            threshold: Confidence threshold for selection
        """
        self.batch_size = batch_size
        self.threshold = threshold
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            use_fast=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        
        # Load model
        model_kwargs = {
            "device_map": "auto" if device is None else device,
            "torch_dtype": torch.bfloat16,
            "cache_dir": cache_dir,
            "max_length": 4096,
        }
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model,
            **model_kwargs
        )
        
        if checkpoint:
            self.peft_config = PeftConfig.from_pretrained(checkpoint)
            self.model = PeftModel.from_pretrained(
                self.base_model,
                checkpoint
            )
        else:
            self.model = self.base_model
            
        # Prepare model
        self.model.eval()
        if hasattr(self.model, 'half'):
            self.model.half()
            
        # Cache device and token IDs
        self.device = next(self.model.parameters()).device
        self.yes_token_id = self.tokenizer.encode(
            "Yes",
            add_special_tokens=False
        )[0]
        self.no_token_id = self.tokenizer.encode(
            "No",
            add_special_tokens=False
        )[0]
        
        # Initialize sentence splitter
        self.nlp = spacy.load(
            "en_core_web_sm",
            disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "ner"]
        )
        self.nlp.enable_pipe("senter")
        
        # Clear GPU memory
        torch.cuda.empty_cache()
    
    @lru_cache(maxsize=1024)
    def _generate_prompt(
        self,
        query: str,
        context: str,
        sentence: str
    ) -> str:
        """Generate prompt for relevance classification."""
        return (
            f'<start_of_turn>user\n'
            f'Query:\n{query}\n'
            f'Full context:\n{context}\n'
            f'Sentence:\n{sentence}\n'
            f'Is this sentence useful in answering the query? '
            f'Answer only "Yes" or "No".<end_of_turn>\n'
            f'<start_of_turn>model\n'
        )
    
    def _predict_batch(
        self,
        queries: List[str],
        contexts: List[str],
        sentences: List[str]
    ) -> Tuple[List[str], torch.Tensor]:
        """Predict relevance for a batch of sentences."""
        prompts = [
            self._generate_prompt(query, context, sentence)
            for query, context, sentence
            in zip(queries, contexts, sentences)
        ]
        
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            inputs = self.tokenizer(
                prompts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=4096,
                return_attention_mask=True
            )
            
            inputs = {
                k: v.to(self.device, non_blocking=True)
                for k, v in inputs.items()
            }
            
            with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = self.model(**inputs)
                
                next_token_logits = outputs.logits[:, -1, :]
                relevant_logits = torch.stack([
                    next_token_logits[:, self.yes_token_id],
                    next_token_logits[:, self.no_token_id]
                ], dim=1)
                
                probs = torch.softmax(relevant_logits, dim=1)
                predictions = [
                    "Yes" if p else "No"
                    for p in probs.argmax(dim=1).cpu().numpy()
                ]
        
        return predictions, probs
    
    def _compress_document(
        self,
        query: str,
        document: SearchResult
    ) -> str:
        """Compress a single document by selecting relevant sentences.
        
        Args:
            query: Input question
            document: Single document to compress
            
        Returns:
            Compressed text for the document
        """
        # Combine title and text for full document context
        full_text = f"{document.title}\n{document.text}" if document.title else document.text
        
        # Split into sentences using spaCy
        raw_sentences = [sent.text.strip() for sent in self.nlp(full_text).sents]
        
        # Clean up sentences - remove extra whitespace and empty sentences
        sentences = []
        for sent in raw_sentences:
            # Replace multiple whitespace characters with single space
            cleaned_sent = " ".join(sent.split())
            if cleaned_sent:  # Only keep non-empty sentences
                sentences.append(cleaned_sent)
        
        if not sentences:
            return ""
        
        # Process sentences in batches
        selected_sentences = []
        
        for i in range(0, len(sentences), self.batch_size):
            batch_sentences = sentences[i:i + self.batch_size]
            batch_size = len(batch_sentences)
            
            # Create batch inputs (per-document context)
            queries = [query] * batch_size
            contexts = [full_text] * batch_size  # Use document's own context
            
            # Get predictions for batch
            predictions, probs = self._predict_batch(queries, contexts, batch_sentences)
            
            # Select sentences above threshold
            for j, (sentence, prob) in enumerate(zip(batch_sentences, probs)):
                if prob[0].item() >= self.threshold:  # prob[0] is "Yes" probability
                    selected_sentences.append(sentence)
        
        # Join sentences within document with spaces
        return " ".join(selected_sentences)

    def compress(
        self,
        query: str,
        documents: List[SearchResult]
    ) -> List[SearchResult]:
        """Compress documents using context-aware extraction.
        
        Args:
            query: Input question
            documents: List of documents to compress
            
        Returns:
            List containing single SearchResult with compressed text
        """
        compressed_docs = []
        
        # Compress each document individually
        for doc in documents:
            compressed_text = self._compress_document(query, doc)
            if compressed_text.strip():  # Only keep non-empty compressed documents
                compressed_docs.append(compressed_text)
        
        # Combine all compressed documents with newline separators
        final_compressed_text = "\n".join(compressed_docs)
        
        # Return compressed result
        return [SearchResult(
            evi_id=0,
            docid=0,
            title="",
            text=final_compressed_text,
            score=1.0
        )]