"""
Shared asynchronous runner for DSPy experiments.
Decouples concurrency logic from Streamlit for use in terminal scripts.
"""

import time
import asyncio
import logging
import dspy
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)

# Setup logging
logger = logging.getLogger(__name__)
# Configure if not already configured
if not logger.handlers:
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
    """
    # Run in thread to avoid blocking event loop
    loop = asyncio.get_event_loop()
    
    def _sync_call():
        # CRITICAL: Clear LM history before each call to prevent contamination
        if hasattr(lm, 'history'):
            lm.history = []
        
        with dspy.context(lm=lm):
            try:
                result = extractor_instance(text)
            except Exception as e:
                # Log error for debugging but re-raise for retry
                logger.warning(f"DSPy call failed: {e}")
                raise e
        
        # CRITICAL: Capture history IMMEDIATELY after call
        captured_history = None
        token_usage = None
        
        if hasattr(lm, 'history') and lm.history:
            last_interaction = lm.history[-1].copy() if lm.history else {}
            captured_history = {
                'messages': last_interaction.get('messages', []),
                'response': last_interaction.get('response', None),
                'outputs': last_interaction.get('outputs', [])
            }
            
            # Extract token usage
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
    
    return await loop.run_in_executor(None, _sync_call)


async def run_single_sample_async(
    semaphore, extractor_instance, lm, sample, idx, progress_callback=None
):
    """
    Process a single NER sample with concurrency control.
    """
    async with semaphore:  # Acquire semaphore slot
        try:
            start_time = time.time()
            
            # Call with retry logic
            pred, captured_history, token_usage = await call_dspy_with_retry(extractor_instance, lm, sample['text'])
            
            latency = time.time() - start_time
            
            # Build history dict
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
                if asyncio.iscoroutinefunction(progress_callback):
                    await progress_callback(idx)
                else:
                    progress_callback(idx)
            
            return idx, pred, latency, history, token_usage
            
        except Exception as e:
            logger.error(f"Failed to process sample {idx}: {e}")
            # Return empty result on failure to keep indices aligned
            return idx, {'PER': [], 'ORG': [], 'LOC': [], 'MISC': []}, 0.0, {
                'text': sample['text'],
                'prediction': {},
                'error': str(e)
            }, None


async def process_samples_concurrently(
    test_samples, 
    extractor_instance, 
    lm, 
    max_concurrent=10, 
    progress_callback=None
):
    """
    Process a list of samples concurrently using DSPy.
    
    Args:
        test_samples: List of sample dicts (must have 'text' key)
        extractor_instance: Initialized DSPy module
        lm: Initialized Language Model
        max_concurrent: Max concurrent requests
        progress_callback: Optional callback(idx) for progress updates
        
    Returns:
        tuple: (predictions, latencies, histories, token_usages)
        All lists are sorted by original sample index.
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    
    tasks = [
        run_single_sample_async(
            semaphore, extractor_instance, lm, sample, i, progress_callback
        )
        for i, sample in enumerate(test_samples)
    ]
    
    # Run all tasks
    results = await asyncio.gather(*tasks, return_exceptions=False)
    
    # Sort by index
    results = sorted(results, key=lambda x: x[0])
    
    # Unpack
    predictions = [r[1] for r in results]
    latencies = [r[2] for r in results]
    histories = [r[3] for r in results]
    token_usages = [r[4] for r in results]
    
    return predictions, latencies, histories, token_usages
