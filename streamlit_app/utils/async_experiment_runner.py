"""
Asynchronous experiment runner for concurrent NER API execution.
Uses asyncio with semaphore-based rate limiting and exponential backoff retries.
"""

import time
import asyncio
import logging
import streamlit as st
import dspy
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)

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

# Setup logging for retry attempts
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


from src.utils.async_runner import (
    calculate_safe_concurrency,
    process_samples_concurrently
)

async def run_single_sample_async(*args, **kwargs):
    """Deprecated: Logic moved to src.utils.async_runner"""
    pass


async def run_dspy_model_async(
    test_samples, evaluator, model_name, model_config, use_implicit, use_cot, use_fewshot
):
    """
    Run DSPy model experiment with concurrent API calls.
    
    Uses semaphore-based rate limiting to prevent overwhelming OpenAI API.
    Automatically retries on rate limit errors with exponential backoff.
    
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
    # Calculate safe concurrency level
    # Assuming tier 2 (RPM=500) and 2s average request duration
    max_concurrent = calculate_safe_concurrency(rpm_limit=500, avg_request_duration=2.0)
    
    logger.info(f"Using {max_concurrent} concurrent requests for {model_name}")
    
    with st.spinner(f"Running DSPy with {model_name} ({max_concurrent} concurrent)..."):
        lm = get_lm(model_name)
        
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
        
        # Progress tracking
        progress_bar = st.progress(0)
        progress_text = st.empty()
        completed = 0
        
        def update_progress(idx):
            nonlocal completed
            completed += 1
            progress_text.text(f"Processing sample {completed}/{len(test_samples)}...")
            progress_bar.progress(completed / len(test_samples))
        
        # Run concurrently using shared utility
        predictions, latencies, histories, token_usages = await process_samples_concurrently(
            test_samples,
            extractor_instance,
            lm,
            max_concurrent=max_concurrent,
            progress_callback=update_progress
        )
        
        # Calculate prompt caching statistics
        total_prompt_tokens = sum(t['prompt_tokens'] for t in token_usages if t)
        total_cached_tokens = sum(t['cached_tokens'] for t in token_usages if t)
        total_completion_tokens = sum(t['completion_tokens'] for t in token_usages if t)
        
        cache_hit_rate = (total_cached_tokens / total_prompt_tokens * 100) if total_prompt_tokens > 0 else 0
        
        # Log caching metrics
        logger.info(f"🔍 Prompt Caching Stats:")
        logger.info(f"   Total prompt tokens: {total_prompt_tokens}")
        logger.info(f"   Cached tokens: {total_cached_tokens}")
        logger.info(f"   Cache hit rate: {cache_hit_rate:.1f}%")
        logger.info(f"   Completion tokens: {total_completion_tokens}")
        
        # Display caching info to user
        if total_cached_tokens > 0:
            st.info(f"💰 Prompt caching active! {cache_hit_rate:.1f}% cache hit rate ({total_cached_tokens:,} tokens cached)")
        
        progress_bar.empty()
        progress_text.empty()
        
        # Evaluate results
        eval_results = evaluator.evaluate_model(
            extractor_name,
            predictions,
            model_config,
            latencies=latencies
        )
    
    st.success(f"✅ DSPy extraction complete ({max_concurrent} concurrent requests)")
    return eval_results, predictions, latencies, histories, extractor_name


# Keep synchronous versions for baselines (already fast, no need for async)
def run_regex_baseline(test_samples, evaluator):
    """Run Regex baseline experiment (synchronous)."""
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
    """Run spaCy baseline experiment (synchronous)."""
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
