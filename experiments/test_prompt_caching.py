"""
Terminal script to test and verify OpenAI prompt caching with DSPy.
Runs a small experiment and shows detailed token usage including cached tokens.
"""

import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

import dspy
import json
from src.config import get_lm, MODELS
from src.modules.entity_extractor_implicit import (
    NERExtractorCoTFewShot,
    NERExtractorCoT,
    NERExtractorFewShot
)
from evaluation.multi_sentence_metrics import MultiSentenceNEREvaluator


def load_test_samples(n=10):
    """Load n test samples from the implicit NER dataset."""
    with open('src/data/ner_multi_sentence_samples.json', 'r') as f:
        data = json.load(f)
    return data[:n]


def extract_token_usage(lm_history):
    """Extract detailed token usage from LM history."""
    if not lm_history:
        return None
    
    last_interaction = lm_history[-1]
    
    # DSPy creates an empty usage dict, but actual usage is in response.usage
    response = last_interaction.get('response', None)
    
    if not response:
        print(f"    No response object found")
        return None
    
    # Extract usage from response object
    if hasattr(response, 'usage'):
        usage = response.usage
        print(f"    Found response.usage: {usage}")
        
        # Convert to dict
        token_info = {
            'prompt_tokens': getattr(usage, 'prompt_tokens', 0),
            'completion_tokens': getattr(usage, 'completion_tokens', 0),
            'total_tokens': getattr(usage, 'total_tokens', 0),
            'cached_tokens': 0
        }
        
        # Check for cached tokens in prompt_tokens_details
        if hasattr(usage, 'prompt_tokens_details'):
            details = usage.prompt_tokens_details
            print(f"    Found prompt_tokens_details: {details}")
            if hasattr(details, 'cached_tokens'):
                token_info['cached_tokens'] = details.cached_tokens
        
        return token_info
    else:
        print(f"    response has no usage attribute")
        print(f"    response type: {type(response)}")
        print(f"    response dir: {[x for x in dir(response) if not x.startswith('_')]}")
        return None


def run_caching_test():
    """Run a test to verify prompt caching is working."""
    print("="*80)
    print("OpenAI Prompt Caching Test with DSPy")
    print("="*80)
    print()
    
    # CRITICAL: Enable token usage tracking globally
    dspy.configure(track_usage=True)
    
    # Load test samples - 5 samples to test caching on subsequent requests
    print("Loading test samples...")
    test_samples = load_test_samples(n=5)
    print(f"[OK] Loaded {len(test_samples)} samples")
    print()
    
    # Initialize LM and extractor with CoT + Few-Shot (should trigger caching)
    print("Initializing GPT-4o-mini with CoT + Few-Shot...")
    lm = get_lm('gpt-4o-mini')
    
    # CRITICAL: Disable DSPy's internal cache to get fresh OpenAI responses with usage data
    lm.cache = False
    
    extractor = NERExtractorCoTFewShot()
    print("[OK] Extractor ready")
    print()
    
    # Process samples and collect token usage
    print("Processing samples and tracking token usage...")
    print("-"*80)
    print(f"{'Sample':<8} {'Prompt':<10} {'Cached':<10} {'Completion':<12} {'Status'}")
    print("-"*80)
    
    token_usages = []
    
    for i, sample in enumerate(test_samples):
        try:
            # Clear history before call
            if hasattr(lm, 'history'):
                lm.history = []
            
            # Run extraction
            with dspy.context(lm=lm):
                result = extractor(sample['text'])
            
            # Access usage from lm.history
            if hasattr(lm, 'history') and lm.history:
                last_call = lm.history[-1]
                
                # Debug: print everything about the response
                response = last_call.get('response', None)
                if response:
                    # Dump the entire model to see all fields
                    dumped = response.model_dump()
                    
                    # Check if usage exists and has data
                    usage_from_dump = dumped.get('usage')
                    if usage_from_dump and isinstance(usage_from_dump, dict) and usage_from_dump:
                        token_info = {
                            'prompt_tokens': usage_from_dump.get('prompt_tokens', 0),
                            'completion_tokens': usage_from_dump.get('completion_tokens', 0),
                            'total_tokens': usage_from_dump.get('total_tokens', 0),
                            'cached_tokens': 0
                        }
                        
                        # Check for cached tokens
                        prompt_details = usage_from_dump.get('prompt_tokens_details', {})
                        if prompt_details:
                            token_info['cached_tokens'] = prompt_details.get('cached_tokens', 0)
                        
                        token_usages.append(token_info)
                        
                        cache_status = "<<< CACHE HIT!" if token_info['cached_tokens'] > 0 else ""
                        print(f"{i+1:<8} {token_info['prompt_tokens']:<10} "
                              f"{token_info['cached_tokens']:<10} "
                              f"{token_info['completion_tokens']:<12} {cache_status}")
                        continue
                    
                    # If we got here, couldn't find usage
                    print(f"{i+1:<8} {'N/A':<10} {'N/A':<10} {'N/A':<12} USAGE EMPTY")
                else:
                    print(f"{i+1:<8} {'N/A':<10} {'N/A':<10} {'N/A':<12} NO RESPONSE")
            else:
                print(f"{i+1:<8} {'N/A':<10} {'N/A':<10} {'N/A':<12} NO HISTORY")
                
        except Exception as e:
            print(f"[Sample {i+1}] ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    print("-"*80)
    print()
    
    # Calculate and display statistics
    if token_usages:
        total_prompt = sum(t['prompt_tokens'] for t in token_usages)
        total_cached = sum(t['cached_tokens'] for t in token_usages)
        total_completion = sum(t['completion_tokens'] for t in token_usages)
        
        cache_hit_rate = (total_cached / total_prompt * 100) if total_prompt > 0 else 0
        
        print("Token Usage Summary:")
        print(f"   Total prompt tokens:     {total_prompt:,}")
        print(f"   Total cached tokens:     {total_cached:,} ({cache_hit_rate:.1f}% cache hit rate)")
        print(f"   Total completion tokens: {total_completion:,}")
        print()
        
        if total_cached > 0:
            # Calculate cost savings (GPT-4o-mini pricing)
            input_cost_per_1k = 0.00015
            cached_discount = 0.50  # 50% discount on cached tokens
            
            normal_cost = (total_prompt * input_cost_per_1k / 1000)
            cached_cost = ((total_prompt - total_cached) * input_cost_per_1k / 1000 + 
                          total_cached * input_cost_per_1k * cached_discount / 1000)
            savings = normal_cost - cached_cost
            savings_pct = (savings / normal_cost * 100) if normal_cost > 0 else 0
            
            print("Cost Analysis (input tokens only):")
            print(f"   Without caching: ${normal_cost:.6f}")
            print(f"   With caching:    ${cached_cost:.6f}")
            print(f"   Savings:         ${savings:.6f} ({savings_pct:.1f}%)")
            print()
            print("SUCCESS: Prompt caching is WORKING!")
        else:
            print("WARNING: No cached tokens detected.")
            print()
            print("Possible reasons:")
            print("   1. Prompts < 1024 tokens (caching requires 1024+ token prefix)")
            print("   2. This was the first batch (cache builds after first request)")
            print("   3. Cache expired (5-10 min inactivity timeout)")
            print()
            print("Try running with more samples or check prompt length.")
    
    print()
    print("="*80)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run optimisations')
    parser.add_argument('--model', default='gpt-4o-mini',
                       choices=list(MODELS.keys()),
                       help='Model to use for DSPy')
    args = parser.parse_args()
    if args.model != 'gpt-4o-mini':
        print("No model should be selected, this test is configured to show OpenAIs caching with gpt-4o-mini")
        exit()
    run_caching_test()
    