"""
Experiment runner for NER baselines and DSPy models.
Handles execution of Regex, spaCy, and DSPy extractors with progress tracking.
"""

import time
import streamlit as st
import dspy

from src.baselines.regex_ner import extract_entities_regex
from src.baselines.spacy_ner import extract_entities_spacy
from src.modules.entity_extractor import NERExtractor
from src.modules.entity_extractor_implicit import (
    NERExtractorImplicit,
    NERExtractorCoT,
    NERExtractorFewShot,
    NERExtractorCoTFewShot
)
from src.config import get_lm


def run_regex_baseline(test_samples, evaluator):
    """Run Regex baseline experiment."""
    with st.spinner("Running Regex baseline..."):
        predictions = []
        latencies = []
        progress_bar = st.progress(0)
        
        for i, sample in enumerate(test_samples):
            start_time = time.time()
            pred = extract_entities_regex(sample['text'])
            latencies.append(time.time() - start_time)
            predictions.append(pred)
            progress_bar.progress((i + 1) / len(test_samples))
        
        progress_bar.empty()
        
        results = evaluator.evaluate_model(
            'Regex Baseline',
            predictions,
            {'cost_per_1k_input': 0, 'cost_per_1k_output': 0},
            latencies=latencies
        )
    
    st.success("✅ Regex baseline complete")
    return results, predictions, latencies


def run_spacy_baseline(test_samples, evaluator):
    """Run spaCy baseline experiment."""
    with st.spinner("Running spaCy baseline..."):
        predictions = []
        latencies = []
        progress_bar = st.progress(0)
        
        for i, sample in enumerate(test_samples):
            start_time = time.time()
            pred = extract_entities_spacy(sample['text'])
            latencies.append(time.time() - start_time)
            predictions.append(pred)
            progress_bar.progress((i + 1) / len(test_samples))
        
        progress_bar.empty()
        
        results = evaluator.evaluate_model(
            'spaCy',
            predictions,
            {'cost_per_1k_input': 0, 'cost_per_1k_output': 0},
            latencies=latencies
        )
    
    st.success("✅ spaCy baseline complete")
    return results, predictions, latencies


def run_dspy_model(test_samples, evaluator, model_name, model_config, use_implicit, use_cot, use_fewshot):
    """
    Run DSPy model experiment.
    
    Args:
        test_samples: list of test samples
        evaluator: ModelEvaluator instance
        model_name: str - Model name (e.g., 'gpt-4o-mini')
        model_config: dict - Model pricing configuration
        use_implicit: bool - Whether implicit mode is enabled
        use_cot: bool - Whether to use Chain-of-Thought
        use_fewshot: bool - Whether to use Few-Shot examples
    
    Returns:
        tuple: (results, predictions, latencies, histories, extractor_name)
    """
    with st.spinner(f"Running DSPy with {model_name}..."):
        lm = get_lm(model_name)
        predictions = []
        latencies = []
        histories = []  # Store LLM interaction history
        
        # Select appropriate extractor based on mode and options
        if use_implicit:
            if use_cot and use_fewshot:
                extractor_instance = NERExtractorCoTFewShot()
                extractor_name = "DSPy (CoT + Few-Shot)"
            elif use_cot:
                extractor_instance = NERExtractorCoT()
                extractor_name = "DSPy (CoT)"
            elif use_fewshot:
                extractor_instance = NERExtractorFewShot()
                extractor_name = "DSPy (Few-Shot)"
            else:
                extractor_instance = NERExtractorImplicit()
                extractor_name = "DSPy (Implicit-aware)"
        else:
            extractor_instance = NERExtractor()
            extractor_name = f"DSPy ({model_name})"
        
        progress_bar = st.progress(0)
        progress_text = st.empty()
        
        for i, sample in enumerate(test_samples):
            progress_text.text(f"Processing sample {i+1}/{len(test_samples)}...")
            with dspy.context(lm=lm):
                start_time = time.time()
                pred = extractor_instance(sample['text'])
                latencies.append(time.time() - start_time)
                predictions.append(pred)
                
                # Capture the actual LLM prompt and response from lm.history
                if hasattr(lm, 'history') and lm.history:
                    last_interaction = lm.history[-1]
                    histories.append({
                        'text': sample['text'],
                        'prediction': pred,
                        'messages': last_interaction.get('messages', []),
                        'response': last_interaction.get('response', None),
                        'outputs': last_interaction.get('outputs', [])
                    })
                else:
                    histories.append({
                        'text': sample['text'],
                        'prediction': pred,
                        'messages': [],
                        'response': None,
                        'outputs': []
                    })
            
            progress_bar.progress((i + 1) / len(test_samples))
        
        progress_bar.empty()
        progress_text.empty()
        
        results = evaluator.evaluate_model(
            extractor_name,
            predictions,
            model_config,
            latencies=latencies
        )
    
    st.success("✅ DSPy extraction complete")
    return results, predictions, latencies, histories, extractor_name
