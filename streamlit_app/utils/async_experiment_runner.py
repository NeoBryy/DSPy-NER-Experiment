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


def calculate_safe_concurrency(rpm_limit=500, avg_request_duration=2.0):
    """
    Calculate safe concurrency based on OpenAI rate limits.
    
    Formula: safe_concurrency = (RPM_limit × avg_request_duration_seconds) / 60
    
    Args:
        rpm_limit: Requests per minute limit (default 500 for tier 2)
        avg_request_duration: Average request duration in seconds (default 2s)
    
    Returns:
        int: Safe number of concurrent requests (capped at 10)
    """
    calculated = int((rpm_limit * avg_request_duration) / 60)
    # Cap at 10 based on production observations (degraded performance above 8-10)
    return min(calculated, 10)


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((Exception,)),  # Retry on rate limits and server errors
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True
)
async def call_dspy_with_retry(extractor_instance, lm, text):
    """
    Call DSPy extractor with automatic retry on failures.
    
    Uses exponential backoff: 1s, 2s, 4s, 8s, 10s (max)
    Retries up to 5 times on rate limit (429) or server errors (5xx)
    
    Args:
        extractor_instance: DSPy extractor module
        lm: Language model instance
        text: Input text to process
    
    Returns:
        tuple: (prediction, captured_history)
    """
   # Run in thread to avoid blocking event loop
    loop = asyncio.get_event_loop()
    
    def _sync_call():
        # CRITICAL: Clear LM history before each call to prevent contamination
        # across concurrent requests. Without this, concurrent calls share history
        # and LLM outputs include entities from other samples!
        if hasattr(lm, 'history'):
            lm.history = []
        
        with dspy.context(lm=lm):
            result = extractor_instance(text)
        
        # CRITICAL: Capture history IMMEDIATELY after call, while still in this thread
        # If we capture outside the thread, other concurrent tasks may have already
        # polluted lm.history with their data, causing wrong prompts to display
        captured_history = None
        token_usage = None
        
        if hasattr(lm, 'history') and lm.history:
            last_interaction = lm.history[-1].copy() if lm.history else {}
            captured_history = {
                'messages': last_interaction.get('messages', []),
                'response': last_interaction.get('response', None),
                'outputs': last_interaction.get('outputs', [])
            }
            
            # Extract token usage including cached tokens for prompt caching analysis
            response_obj = last_interaction.get('response', None)
            if response_obj and hasattr(response_obj, 'usage'):
                usage = response_obj.usage
                token_usage = {
                    'prompt_tokens': usage.prompt_tokens if hasattr(usage, 'prompt_tokens') else 0,
                    'completion_tokens': usage.completion_tokens if hasattr(usage, 'completion_tokens') else 0,
                    'total_tokens': usage.total_tokens if hasattr(usage, 'total_tokens') else 0,
                    'cached_tokens': 0  # Default to 0
                }
                
                # Check for cached tokens in prompt_tokens_details
                if hasattr(usage, 'prompt_tokens_details'):
                    details = usage.prompt_tokens_details
                    if hasattr(details, 'cached_tokens'):
                        token_usage['cached_tokens'] = details.cached_tokens
        
        return result, captured_history, token_usage
    
    result, captured_history, token_usage = await loop.run_in_executor(None, _sync_call)
    return result, captured_history, token_usage


async def run_single_sample_async(
    semaphore, extractor_instance, lm, sample, idx, progress_callback=None
):
    """
    Process a single NER sample with concurrency control.
    
    Args:
        semaphore: asyncio.Semaphore to limit concurrent requests
        extractor_instance: DSPy extractor module
        lm: Language model instance
        sample: Test sample dict with 'text' field
        idx: Sample index
        progress_callback: Optional callback(idx) called on completion
    
    Returns:
        tuple: (idx, prediction, latency, history, token_usage)
    """
    async with semaphore:  # Acquire semaphore slot
        try:
            start_time = time.time()
            
            # Call with retry logic - now returns prediction, history, and token usage
            pred, captured_history, token_usage = await call_dspy_with_retry(extractor_instance, lm, sample['text'])
            
            latency = time.time() - start_time
            
            # Build history dict with captured data
            history = {
                'text': sample['text'],
                'prediction': pred,
                'messages': [],
                'response': None,
                'outputs': []
            }
            
            if captured_history:
                history['messages'] = captured_history.get('messages', [])
                history['response'] = captured_history.get('response', None)
                history['outputs'] = captured_history.get('outputs', [])
            
            # Call progress callback if provided
            if progress_callback:
                progress_callback(idx)
            
            return idx, pred, latency, history, token_usage
            
        except Exception as e:
            logger.error(f"Failed to process sample {idx} after retries: {e}")
            # Return empty result on failure
            return idx, {'PER': [], 'ORG': [], 'LOC': [], 'MISC': []}, 0.0, {
                'text': sample['text'],
                'prediction': {},
                'error': str(e)
            }, None  # No token usage on failure


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
        
        # Create semaphore for rate limiting
        semaphore = asyncio.Semaphore(max_concurrent)
        
        # Progress tracking
        progress_bar = st.progress(0)
        progress_text = st.empty()
        completed = 0
        
        def update_progress(idx):
            nonlocal completed
            completed += 1
            progress_text.text(f"Processing sample {completed}/{len(test_samples)}...")
            progress_bar.progress(completed / len(test_samples))
        
        # Create tasks for all samples
        tasks = [
            run_single_sample_async(
                semaphore, extractor_instance, lm, sample, i, update_progress
            )
            for i, sample in enumerate(test_samples)
        ]
        
        # Run all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=False)
        
        # Sort results by index to maintain order
        results = sorted(results, key=lambda x: x[0])
        
        # Extract components
        predictions = [r[1] for r in results]
        latencies = [r[2] for r in results]
        histories = [r[3] for r in results]
        token_usages = [r[4] for r in results]  # Extract token usage
        
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
