"""
Evaluation metrics for Named Entity Recognition.
Calculates Precision, Recall, and F1 scores for entity extraction.
"""

from typing import List, Dict


class ModelEvaluator:
    """Evaluates NER model performance."""
    
    def __init__(self, test_data):
        """
        Initialize evaluator with test data.
        
        Args:
            test_data: List of dicts with 'text' and 'entities' keys
        """
        self.test_data = test_data
    
    def evaluate_model(self, model_name, predictions, cost_config, latencies=None):
        """
        Evaluate model predictions.
        
        Args:
            model_name: Name of the model
            predictions: List of predicted entity dicts
            cost_config: Dict with cost_per_1k_input and cost_per_1k_output
            latencies: Optional list of latency measurements
        
        Returns:
            Dict with evaluation results
        """
        if latencies is None:
            latencies = []
        
        # Aggregate TP, FP, FN across all samples for each entity type
        entity_types = ['PER', 'ORG', 'LOC', 'MISC']
        tp_counts = {et: 0 for et in entity_types}
        fp_counts = {et: 0 for et in entity_types}
        fn_counts = {et: 0 for et in entity_types}
        
        # Process each sample
        for pred, sample in zip(predictions, self.test_data):
            for entity_type in entity_types:
                pred_set = set(pred.get(entity_type, []))
                true_set = set(sample['entities'].get(entity_type, []))
                
                # Count TP, FP, FN
                tp_counts[entity_type] += len(pred_set & true_set)
                fp_counts[entity_type] += len(pred_set - true_set)
                fn_counts[entity_type] += len(true_set - pred_set)
        
        # Calculate metrics for each entity type
        metrics = {}
        for entity_type in entity_types:
            tp = tp_counts[entity_type]
            fp = fp_counts[entity_type]
            fn = fn_counts[entity_type]
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            metrics[f'{entity_type}_precision'] = precision
            metrics[f'{entity_type}_recall'] = recall
            metrics[f'{entity_type}_f1'] = f1
        
        # Calculate overall micro-averaged metrics (aggregate all entity types)
        tp_total = sum(tp_counts.values())
        fp_total = sum(fp_counts.values())
        fn_total = sum(fn_counts.values())
        
        overall_precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
        overall_recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
        overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
        
        metrics['overall_precision'] = overall_precision
        metrics['overall_recall'] = overall_recall
        metrics['overall_f1'] = overall_f1
        
        # Calculate cost (rough estimate based on text length)
        total_input_tokens = sum(len(sample['text'].split()) * 1.3 for sample in self.test_data)
        total_output_tokens = sum(
            sum(len(entity) for entities in pred.values() for entity in entities) * 1.3
            for pred in predictions
        )
        
        input_cost = (total_input_tokens / 1000) * cost_config['cost_per_1k_input']
        output_cost = (total_output_tokens / 1000) * cost_config['cost_per_1k_output']
        total_cost = input_cost + output_cost
        
        # Calculate latency stats
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        total_latency = sum(latencies) if latencies else 0
        
        return {
            'model_name': model_name,
            'metrics': metrics,
            'estimated_cost': total_cost,
            'avg_latency': avg_latency,
            'total_latency': total_latency,
            'num_samples': len(self.test_data)
        }


if __name__ == '__main__':
    # Test the metrics
    test_data = [
        {
            'text': 'Tim Cook works at Apple Inc.',
            'entities': {'PER': ['Tim Cook'], 'ORG': ['Apple Inc.'], 'LOC': [], 'MISC': []}
        },
        {
            'text': 'Elon Musk founded Tesla.',
            'entities': {'PER': ['Elon Musk'], 'ORG': ['Tesla'], 'LOC': [], 'MISC': []}
        }
    ]
    
    predictions = [
        {'PER': ['Tim Cook'], 'ORG': ['Apple Inc.'], 'LOC': [], 'MISC': []},  # Perfect
        {'PER': ['Elon Musk'], 'ORG': [], 'LOC': [], 'MISC': ['Tesla']}  # Wrong type for Tesla
    ]
    
    evaluator = ModelEvaluator(test_data)
    results = evaluator.evaluate_model(
        'Test',
        predictions,
        {'cost_per_1k_input': 0, 'cost_per_1k_output': 0}
    )
    
    print("Test metrics:")
    for key, value in results['metrics'].items():
        print(f"  {key}: {value:.3f}")
    print("\nExpected:")
    print("  PER: P=1.0, R=1.0, F1=1.0 (2/2 correct)")
    print("  ORG: P=1.0, R=0.5, F1=0.667 (1/2 correct, 1 missed)")
    print("  MISC: P=0.0, R=0.0, F1=0.0 (1 wrong prediction)")
