#!/usr/bin/env python3

from typing import List
import string
import unicodedata
import regex
import spacy
from collections import Counter
from typing import List, Union, Any, Dict
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Load spaCy model for sentence segmentation
nlp = spacy.load("en_core_web_sm", exclude=['tagger', 'parser', 'ner', 'lemmatizer', 'tok2vec'])
nlp.enable_pipe("senter")

def count_sentences(text: str, nlp_model=None) -> int:
    """Count the number of sentences in a text using spaCy."""
    model = nlp_model if nlp_model else nlp
    doc = model(text)
    return len(list(doc.sents))

def get_words(text: str) -> List[str]:
    """Extract words from text using spaCy tokenization."""
    doc = nlp(text)
    return [token.text.lower() for token in doc if token.is_alpha]


def normalize_answer(s: str) -> str:
    """Normalize answer using DPR-style normalization (improved from SQuAD)."""
    def remove_articles(text):
        return regex.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    def normalize_unicode(text):
        return unicodedata.normalize('NFD', text)

    return white_space_fix(remove_articles(remove_punc(lower(normalize_unicode(s)))))


def f1_score(prediction: str, ground_truth: str) -> float:
    """Calculate F1 score between prediction and a single ground truth."""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens) if len(prediction_tokens) > 0 else 0
    recall = 1.0 * num_same / len(ground_truth_tokens) if len(ground_truth_tokens) > 0 else 0
    if precision + recall == 0:
        return 0
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction: str, ground_truth: str) -> bool:
    """Calculate exact match score between prediction and a single ground truth."""
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def multi_answer_em(prediction: str, ground_truths: Union[str, List[str]]) -> float:
    """Calculate EM score for multiple ground truths - returns max EM."""
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    return max([exact_match_score(prediction, gt) for gt in ground_truths])


def multi_answer_f1(prediction: str, ground_truths: Union[str, List[str]]) -> float:
    """Calculate F1 score for multiple ground truths - returns max F1."""
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    return max([f1_score(prediction, gt) for gt in ground_truths])


def evaluate_predictions(predictions: List[str], ground_truths: List[Any]) -> dict:
    """
    Evaluate a list of predictions against ground truths.
    
    Args:
        predictions: List of predicted answers
        ground_truths: List of ground truth answers (can be strings or lists of strings)
    
    Returns:
        Dictionary with evaluation metrics
    """
    if len(predictions) != len(ground_truths):
        raise ValueError(f"Predictions ({len(predictions)}) and ground truths ({len(ground_truths)}) must have same length")
    
    total_em = 0
    total_f1 = 0
    
    for pred, gt in zip(predictions, ground_truths):
        total_em += multi_answer_em(pred, gt)
        total_f1 += multi_answer_f1(pred, gt)
    
    count = len(predictions)
    
    return {
        'exact_match': total_em,
        'f1': total_f1,
        'count': count,
        'exact_match_percentage': 100.0 * total_em / count if count > 0 else 0,
        'f1_percentage': 100.0 * total_f1 / count if count > 0 else 0
    }


def print_evaluation_results(results: dict, title: str = "EVALUATION RESULTS"):
    """Print evaluation results in a formatted way."""
    print("\n" + "="*80)
    print(title)
    print("="*80)
    print(f"Total Questions: {results['count']}")
    print(f"Exact Match: {results['exact_match_percentage']:.2f}%")
    print(f"F1 Score: {results['f1_percentage']:.2f}%")
    print("="*80)


def format_prompt(question: str, context: str, tokenizer=None, use_api: bool = False) -> str:
    """Format prompt for both local model and API."""
    if use_api:
        # Simple format for API
        return f"Context information is: ```{context}```\n\nGiven provided context (might not be sufficient for below query), answer the query without any explanation.\nQuery: `{question}`\nAnswer (in plain text):"
    else:
        # Chat template format for local model
        messages = [
            {"role": "user", "content": f"Context information is: ```{context}```\n\nGiven provided context (might not be sufficient for below query), answer the query without any explanation.\nQuery: `{question}`\nAnswer (in plain text):"}
        ]
        
        return tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True,
        )

def generate_answers_api(prompts: List[str], api_model: str, max_workers: int = 8, max_output_tokens: int = 360) -> List[str]:
    """Generate answers using Vertex AI API with parallel processing."""
    from ask_vertex import ask_vertex
    
    def call_api_single(prompt):
        try:
            response = ask_vertex(
                prompt,
                model=api_model,
                max_output_tokens=max_output_tokens,
            )
            return response if response is not None else ""
        except Exception as e:
            print(f"Error in API call: {e}")
            return ""
    
    # Use ThreadPoolExecutor for parallel API calls
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        responses = list(tqdm(
            executor.map(call_api_single, prompts),
            total=len(prompts),
            desc="Generating answers via API",
            leave=False,
        ))
    
    return responses

def evaluate_batch(predictions: List[str], ground_truths: List[Any]) -> Dict[str, float]:
    """Evaluate a batch of predictions."""
    results = evaluate_predictions(predictions, ground_truths)
    return {
        'exact_match': results['exact_match'],
        'f1': results['f1'],
        'count': results['count']
    }

def preprocess_contexts(question_data: Dict, k: int, nlp_model=None) -> List[Dict]:
    """Preprocess contexts for a single question."""
    from collections import defaultdict
    
    def find_overlap(text1: str, text2: str):
        """Find overlap between two texts."""
        if not text1 or not text2:
            return None, 0

        MIN_OVERLAP_CHARS = 11

        max_overlap_len = min(len(text1), len(text2))
        
        for overlap_len in range(max_overlap_len, MIN_OVERLAP_CHARS - 1, -1):
            text1_suffix = text1[-overlap_len:]
            text2_prefix = text2[:overlap_len]
            
            if text1_suffix == text2_prefix:
                merged_text = text1 + text2[overlap_len:]
                return merged_text, overlap_len

        for overlap_len in range(max_overlap_len, MIN_OVERLAP_CHARS - 1, -1):
            text2_suffix = text2[-overlap_len:]
            text1_prefix = text1[:overlap_len]
            
            if text2_suffix == text1_prefix:
                merged_text = text2 + text1[overlap_len:]
                return merged_text, overlap_len

        return None, 0

    def merge_overlapping_segments(segments: List[Dict], nlp_model):
        """Merge segments with same title that have overlapping content."""
        title_groups = defaultdict(list)
        for segment in segments:
            title_groups[segment['title']].append(segment)
        
        merged_segments = []
        
        for title, group in title_groups.items():
            if len(group) == 1:
                merged_segments.append(group[0])
                continue
            
            unmerged = list(group)
            
            while len(unmerged) > 1:
                merged_any = False
                
                for i in range(len(unmerged)):
                    for j in range(i + 1, len(unmerged)):
                        seg1, seg2 = unmerged[i], unmerged[j]
                        
                        merged_text, overlap_length = find_overlap(seg1['text'], seg2['text'])
                        
                        if overlap_length > 0:
                            if count_sentences(merged_text, nlp_model) <= 20:
                                merged_segment = {
                                    'title': title,
                                    'text': merged_text,
                                    'score': max(float(seg1['score']), float(seg2['score'])),
                                    'hasanswer': seg1['hasanswer'] or seg2['hasanswer']
                                }
                                
                                unmerged = [s for k, s in enumerate(unmerged) if k != i and k != j]
                                unmerged.append(merged_segment)
                                merged_any = True
                                break
                    
                    if merged_any:
                        break
                
                if not merged_any:
                    break
            
            merged_segments.extend(unmerged)
        
        return merged_segments

    def add_title_prefix(segment: Dict) -> Dict:
        """Add title prefix to segment if title is not in the text."""
        if segment['title'].lower() not in segment['text'].lower():
            segment['text'] = segment['title'] + '. ' + segment['text']
        return segment

    def create_grouped_segment(segments: List[Dict]) -> Dict:
        """Create a single segment from a group of segments."""
        if not segments:
            return None

        combined_text = ""
        for i, segment in enumerate(segments):
            if i > 0:
                if combined_text.rstrip().endswith(('.', '?', '!', ';', ':', ',')):
                    combined_text += "\n"
                else:
                    combined_text += ".\n"
            combined_text += segment['text']

        max_score = max(float(seg['score']) for seg in segments)

        return {
            'title': segments[0]['title'],
            'text': combined_text,
            'score': str(max_score),
            'hasanswer': any(seg['hasanswer'] for seg in segments)
        }

    def group_short_segments(segments: List[Dict], nlp_model) -> List[Dict]:
        """Group short segments together while keeping long segments separate."""
        long_segments = []
        short_segments = []
        
        for segment in segments:
            sentence_count = count_sentences(segment['text'], nlp_model)
            if sentence_count >= 5:
                long_segments.append(segment)
            else:
                short_segments.append(segment)
        
        short_segments.sort(key=lambda x: float(x['score']), reverse=True)
        
        grouped_segments = []
        current_group = []
        current_sentence_count = 0
        
        for segment in short_segments:
            segment = add_title_prefix(segment)
            
            segment_sentences = count_sentences(segment['text'], nlp_model)
            
            if current_sentence_count + segment_sentences <= 15:
                current_group.append(segment)
                current_sentence_count += segment_sentences
            else:
                if current_group:
                    grouped_segments.append(create_grouped_segment(current_group))
                
                current_group = [segment]
                current_sentence_count = segment_sentences
        
        if current_group:
            grouped_segments.append(create_grouped_segment(current_group))
        
        return long_segments + grouped_segments

    # Use provided nlp_model or default one
    model = nlp_model if nlp_model else nlp
    
    # Extract top k contexts
    top_k_contexts = question_data['ctxs'][:k]

    # Step 1: Merge overlapping segments with same title
    merged_segments = merge_overlapping_segments(top_k_contexts, model)

    # Step 2: Group short segments and separate long segments
    processed_segments = group_short_segments(merged_segments, model)
    
    return processed_segments