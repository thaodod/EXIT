"""EXIT implementation for context-aware extractive compression."""

import torch
import spacy
from typing import List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
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
        
        print("Loading EXIT models...")
        
        # Initialize EXIT compression model (following exit_rag.py pattern)
        base_model_obj = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map=device,
            torch_dtype=torch.bfloat16,
            cache_dir=cache_dir
        )
        self.model = PeftModel.from_pretrained(base_model_obj, checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)

        self.model.eval()
        if hasattr(self.model, 'half'):
            self.model.half()
        
        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Cache device
        self.device = next(self.model.parameters()).device
        
        # Initialize sentence splitter
        self.nlp = spacy.load(
            "en_core_web_sm",
            disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "ner"]
        )
        self.nlp.enable_pipe("senter")
        
        # Clear GPU memory
        torch.cuda.empty_cache()
    
    def get_sentence_relevance(
        self,
        query: str,
        context: str,
        sentence: str,
    ) -> Tuple[bool, float]:
        """Single sentence relevance (kept for compatibility)."""
        results = self.get_batch_sentence_relevance(query, context, [sentence])
        return results[0]
    
    def get_batch_sentence_relevance(
        self,
        query: str,
        context: str,
        sentences: List[str],
    ) -> List[Tuple[bool, float]]:
        """Batch process multiple sentences for relevance scoring."""
        if not sentences:
            return []
        
        # Create prompts for all sentences
        prompts = []
        for sentence in sentences:
            prompt = f'''<start_of_turn>user
Query:
{query}
Full context:
{context}
Sentence:
{sentence}
Is this sentence useful in answering the query? Answer only "Yes" or "No".<end_of_turn>
<start_of_turn>model
'''
            prompts.append(prompt)
        
        # Process in batches
        results = []
        for i in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[i:i + self.batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=9216,  # Adjust as needed
            ).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                # Get Yes/No token IDs
                yes_id = self.tokenizer.encode("Yes", add_special_tokens=False)[0]
                no_id = self.tokenizer.encode("No", add_special_tokens=False)[0]
                
                # Process each item in the batch
                for j in range(len(batch_prompts)):
                    # Get logits for the last token of this sequence
                    seq_len = inputs['attention_mask'][j].sum().item() - 1  # -1 for 0-indexing
                    logits = outputs.logits[j, seq_len, [yes_id, no_id]]
                    prob = torch.softmax(logits, dim=0)[0].item()
                    
                    is_relevant = prob >= self.threshold
                    results.append((is_relevant, prob))
        
        return results
    
    def _compress_document(
        self,
        query: str,
        document: SearchResult
    ) -> str:
        """Compress a single document by selecting relevant sentences (per-document as in paper).
        """
        # Combine title and text for full document context
        full_text = f"{document.title}\n{document.text}" if document.title else document.text
        
        # Split into sentences using spaCy (following exit_rag.py pattern)
        sentences = [sent.text.strip() for sent in self.nlp(full_text).sents]
        
        if not sentences:
            return ""
        
        # Get relevance scores for all sentences using batch processing
        relevance_results = self.get_batch_sentence_relevance(
            query,
            full_text,  # Use document's own context (as per paper methodology)
            sentences,
        )
        
        # Select relevant sentences
        selected_sentences = []
        for sent, (is_relevant, score) in zip(sentences, relevance_results):
            if is_relevant:
                selected_sentences.append(sent)
        
        # Join sentences within document with spaces
        return " ".join(selected_sentences)

    def compress(
        self,
        query: str,
        documents: List[SearchResult]
    ) -> List[SearchResult]:
        """Compress documents using context-aware extraction (following paper methodology).
        
        Args:
            query: Input question
            documents: List of documents to compress
            
        Returns:
            List containing single SearchResult with compressed text
        """
        # Sort documents by score in descending order (as per paper)
        sorted_documents = sorted(documents, key=lambda x: x.score, reverse=True)
        
        compressed_docs = []
        total_original_sentences = 0
        total_selected_sentences = 0
        
        # Compress each document individually (as per paper methodology)
        for doc in sorted_documents:
            # Count original sentences
            full_text = f"{doc.title}\n{doc.text}" if doc.title else doc.text
            original_sentences = [sent.text.strip() for sent in self.nlp(full_text).sents]
            total_original_sentences += len(original_sentences)
            
            compressed_text = self._compress_document(query, doc)
            if compressed_text.strip():  # Only keep non-empty compressed documents
                compressed_docs.append(compressed_text)
                # Count selected sentences
                selected_sentences = [sent.text.strip() for sent in self.nlp(compressed_text).sents]
                total_selected_sentences += len(selected_sentences)
        
        # Combine all compressed documents with newline separators (maintaining score order)
        final_compressed_text = "\n".join(compressed_docs)
        
        # print(f"Compressed {total_selected_sentences}/{total_original_sentences} sentences from {len(sorted_documents)} documents")
        # Return compressed result
        return [SearchResult(
            evi_id=0,
            docid=0,
            title="",
            text=final_compressed_text,
            score=1.0
        )]