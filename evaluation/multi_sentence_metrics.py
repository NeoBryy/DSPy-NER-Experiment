"""
Evaluator for multi-sentence NER with separated explicit and implicit metrics.
Measures performance on sentence 1 (explicit) and sentence 2 (implicit) independently.
"""

from collections import defaultdict



from difflib import SequenceMatcher

def compute_similarity(a, b):
    """Compute character-level similarity ratio (0.0 - 1.0) case-insensitive."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def score_predictions_vs_truth(preds, truth_entities, threshold=0.8):
    """
    Greedy matching of predictions to ground truth.
    Returns:
        tp_score: Sum of completion scores for matched pairs
        matched_preds: Set of predictions that were matched
    """
    # Calculate all pairwise scores
    matches = []
    for p in preds:
        for t in truth_entities:
            score = compute_similarity(p, t)
            if score >= threshold:
                matches.append((score, p, t))
    
    # Sort by score descending to be greedy
    matches.sort(key=lambda x: x[0], reverse=True)
    
    tp_score = 0.0
    matched_preds = set()
    matched_truth = set()
    
    for score, p, t in matches:
        if p not in matched_preds and t not in matched_truth:
            tp_score += score
            matched_preds.add(p)
            matched_truth.add(t)
    
    return tp_score, matched_preds

def filter_preds_by_text(preds, text):
    """Return predictions that appear in the text (case-insensitive)."""
    text_lower = text.lower()
    return {p for p in preds if p.lower() in text_lower}
    
    
class MultiSentenceNEREvaluator:
    """
    Evaluates NER performance on multi-sentence samples.
    Separates explicit extraction (sentence 1) from implicit resolution (sentence 2).
    """
    
    def __init__(self, test_data):
        """
        Initialize evaluator with test data.
        
        Args:
            test_data: List of dicts with:
                - 'text': Full text
                - 'sentence1': Explicit sentence
                - 'sentence2': Implicit sentence
                - 'sentence1_entities': Ground truth for sentence 1
                - 'sentence2_entities': Ground truth for sentence 2
                - 'implicit_refs': List of implicit references
        """
        self.test_data = test_data
    
    def evaluate_model(self, model_name, predictions, cost_config, latencies):
        """
        Evaluate model performance with separated explicit/implicit metrics.
        
        Args:
            model_name: Name of the model
            predictions: List of prediction dicts (one per sample)
            cost_config: Dict with 'cost_per_1k_input' and 'cost_per_1k_output'
            latencies: List of latency values (one per sample)
        
        Returns:
            Dict with metrics, cost, and latency information
        """
        
        # Track counts for explicit (sentence 1)
        explicit_tp = defaultdict(int)
        explicit_fp = defaultdict(int)
        explicit_fn = defaultdict(int)
        
        # Track counts for implicit (sentence 2)
        implicit_tp = defaultdict(int)
        implicit_fp = defaultdict(int)
        implicit_fn = defaultdict(int)
        
        # Process each sample


        # Process each sample
        for pred, sample in zip(predictions, self.test_data):
            # 1. Evaluate Explicit (Sentence 1)
            for entity_type in ['PER', 'ORG', 'LOC', 'MISC']:
                all_preds = set(pred.get(entity_type, []))
                sent1_truth = set(sample['sentence1_entities'].get(entity_type, []))
                
                # Filter to preds that actually appear in sentence 1 text
                sent1_preds = filter_preds_by_text(all_preds, sample['sentence1'])
                
                # Calculate Explicit Metrics with Fuzzy Matching
                tp, matched_preds = score_predictions_vs_truth(sent1_preds, sent1_truth)
                
                # FP = (Preds in Sent1) - (Matched Preds) + (Penalty for partial matches)
                # This simplifies to: Total Sent1 Preds - TP Score
                explicit_tp[entity_type] += tp
                explicit_fp[entity_type] += len(sent1_preds) - tp
                
                # FN = (Truth) - (Matched Truth) + (Penalty for partial matches)
                # This simplifies to: Total Truth - TP Score
                explicit_fn[entity_type] += len(sent1_truth) - tp
                
            # 2. Evaluate Implicit (Sentence 2 / Implicit Refs)
            for entity_type in ['PER', 'ORG', 'LOC', 'MISC']:
                all_preds = set(pred.get(entity_type, []))
                
                # Identify implicit candidates: Preds NOT matched to explicit entities
                # Note: We need to re-run explicit match logic or cache it?
                # For simplicity/safety: Filter out preds that appear in Sent1 text
                # This explicitly avoids double-counting Sent1 entities as Implicit FPs
                sent1_preds = filter_preds_by_text(all_preds, sample['sentence1'])
                implicit_candidates = all_preds - sent1_preds
                
                # Get implicit reference texts
                implicit_truth = {ref['text'] for ref in sample.get('implicit_refs', []) if ref['type'] == entity_type}
                
                # Calculate Implicit Metrics with Fuzzy Matching
                tp, matched_preds = score_predictions_vs_truth(implicit_candidates, implicit_truth)
                
                implicit_tp[entity_type] += tp
                implicit_fp[entity_type] += len(implicit_candidates) - tp
                implicit_fn[entity_type] += len(implicit_truth) - tp
        
        # Calculate explicit metrics
        explicit_metrics = {}
        for entity_type in ['PER', 'ORG', 'LOC', 'MISC']:
            tp = explicit_tp[entity_type]
            fp = explicit_fp[entity_type]
            fn = explicit_fn[entity_type]
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            explicit_metrics[f'explicit_{entity_type}_precision'] = precision
            explicit_metrics[f'explicit_{entity_type}_recall'] = recall
            explicit_metrics[f'explicit_{entity_type}_f1'] = f1
            explicit_metrics[f'explicit_{entity_type}_count'] = tp + fn
        
        # Calculate implicit metrics
        implicit_metrics = {}
        for entity_type in ['PER', 'ORG', 'LOC', 'MISC']:
            tp = implicit_tp[entity_type]
            fp = implicit_fp[entity_type]
            fn = implicit_fn[entity_type]
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            implicit_metrics[f'implicit_{entity_type}_precision'] = precision
            implicit_metrics[f'implicit_{entity_type}_recall'] = recall
            implicit_metrics[f'implicit_{entity_type}_f1'] = f1
            implicit_metrics[f'implicit_{entity_type}_count'] = tp + fn
        
        # Overall explicit metrics
        total_explicit_tp = sum(explicit_tp.values())
        total_explicit_fp = sum(explicit_fp.values())
        total_explicit_fn = sum(explicit_fn.values())
        
        overall_explicit_precision = total_explicit_tp / (total_explicit_tp + total_explicit_fp) if (total_explicit_tp + total_explicit_fp) > 0 else 0
        overall_explicit_recall = total_explicit_tp / (total_explicit_tp + total_explicit_fn) if (total_explicit_tp + total_explicit_fn) > 0 else 0
        overall_explicit_f1 = 2 * overall_explicit_precision * overall_explicit_recall / (overall_explicit_precision + overall_explicit_recall) if (overall_explicit_precision + overall_explicit_recall) > 0 else 0
        
        # Overall implicit metrics
        total_implicit_tp = sum(implicit_tp.values())
        total_implicit_fp = sum(implicit_fp.values())
        total_implicit_fn = sum(implicit_fn.values())
        
        overall_implicit_precision = total_implicit_tp / (total_implicit_tp + total_implicit_fp) if (total_implicit_tp + total_implicit_fp) > 0 else 0
        overall_implicit_recall = total_implicit_tp / (total_implicit_tp + total_implicit_fn) if (total_implicit_tp + total_implicit_fn) > 0 else 0
        overall_implicit_f1 = 2 * overall_implicit_precision * overall_implicit_recall / (overall_implicit_precision + overall_implicit_recall) if (overall_implicit_precision + overall_implicit_recall) > 0 else 0
        
        # Combine metrics
        metrics = {
            **explicit_metrics,
            **implicit_metrics,
            'overall_explicit_precision': overall_explicit_precision,
            'overall_explicit_recall': overall_explicit_recall,
            'overall_explicit_f1': overall_explicit_f1,
            'overall_implicit_precision': overall_implicit_precision,
            'overall_implicit_recall': overall_implicit_recall,
            'overall_implicit_f1': overall_implicit_f1,
        }
        
        # Calculate cost (basic estimation)
        total_input_tokens = sum(len(sample['text'].split()) * 1.3 for sample in self.test_data)
        total_output_tokens = sum(len(str(pred).split()) * 1.3 for pred in predictions)
        
        cost = (total_input_tokens / 1000 * cost_config['cost_per_1k_input'] +
                total_output_tokens / 1000 * cost_config['cost_per_1k_output'])
        
        # Calculate latency
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        total_latency = sum(latencies)
        
        return {
            'model_name': model_name,
            'metrics': metrics,
            'estimated_cost': cost,
            'avg_latency': avg_latency,
            'total_latency': total_latency,
            'num_samples': len(self.test_data)
        }
