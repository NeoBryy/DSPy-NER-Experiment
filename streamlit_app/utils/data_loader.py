"""
Data loading utilities for NER experiments.
Handles loading standard and implicit NER datasets.
"""

import json
from evaluation.metrics import ModelEvaluator
from evaluation.multi_sentence_metrics import MultiSentenceNEREvaluator


def load_test_data(use_implicit, sample_size):
    """
    Load appropriate test data based on mode.
    
    Args:
        use_implicit: bool - Whether to load implicit NER data
        sample_size: int - Number of samples to load
    
    Returns:
        tuple: (test_samples, evaluator_class, data_info)
            - test_samples: list of dict
            - evaluator_class: ModelEvaluator or MultiSentenceNEREvaluator
            - data_info: str description
    """
    if use_implicit:
        # Load multi-sentence implicit data
        with open('src/data/ner_multi_sentence_samples.json', 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        evaluator_class = MultiSentenceNEREvaluator
        data_info = "multi-sentence implicit"
    else:
        # Load standard NER data
        with open('src/data/ner_samples.json', 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        evaluator_class = ModelEvaluator
        data_info = "standard"
    
    test_samples = test_data[:sample_size]
    
    return test_samples, evaluator_class, data_info
