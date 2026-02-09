"""
Compare DSPy variants for implicit NER:
1. Baseline (original extractor - not implicit-aware)
2. Implicit-aware (told to extract implicit refs)
3. + Chain-of-Thought
4. + Few-Shot
5. + Both CoT and Few-Shot

Measures impact on implicit resolution performance.
"""

import json
import time
from pathlib import Path
from datetime import datetime

import dspy
from src.config import get_lm, MODELS
from src.modules.entity_extractor import NERExtractor  # Original baseline
from src.modules.entity_extractor_implicit import (
    NERExtractorImplicit,
    NERExtractorCoT,
    NERExtractorFewShot,
    NERExtractorCoTFewShot
)
from src.baselines.regex_ner import extract_entities_regex
from src.baselines.spacy_ner import extract_entities_spacy
from evaluation.multi_sentence_metrics import MultiSentenceNEREvaluator


def load_test_data():
    """Load multi-sentence NER test samples."""
    data_path = Path(__file__).parent.parent / 'src' / 'data' / 'ner_multi_sentence_samples.json'
    with open(data_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def run_experiment(model_name='gpt-4o-mini', num_samples=100):
    """
    Compare all DSPy variants for implicit NER.
    """
    print("="*80)
    print(f"DSPy Implicit NER Comparison Experiment ({model_name})")
    print("="*80)
    print("\nComparing:")
    print("  1. Baseline (original NER)")
    print("  2. Implicit-aware (prompted to extract implicit refs)")
    print("  3. + Chain-of-Thought")
    print("  4. + Few-Shot examples")
    print("  5. + Both CoT AND Few-Shot")
    print("\n" + "="*80)
    
    # Load data
    print(f"\n1. Loading test data...")
    test_data = load_test_data()[:num_samples]
    print(f"   Loaded {len(test_data)} samples")
    
    evaluator = MultiSentenceNEREvaluator(test_data)
    lm = get_lm(model_name)
    
    all_results = {}
    
    # Run each DSPy variant
    variants = [
        ('Baseline (original)', NERExtractor()),
        ('Implicit-aware', NERExtractorImplicit()),
        ('+ CoT', NERExtractorCoT()),
        ('+ Few-Shot', NERExtractorFewShot()),
        ('+ CoT + Few-Shot', NERExtractorCoTFewShot()),
    ]
    
    for variant_name, extractor in variants:
        print(f"\n{'='*80}")
        print(f"Running: {variant_name}")
        print('='*80)
        
        predictions = []
        latencies = []
        
        for i, sample in enumerate(test_data):
            with dspy.context(lm=lm):
                start_time = time.time()
                pred = extractor(sample['text'])
                latencies.append(time.time() - start_time)
                predictions.append(pred)
            
            print(f"\r   Processed {i + 1}/{len(test_data)} samples...", end='', flush=True)
        print()
        
        results = evaluator.evaluate_model(
            variant_name,
            predictions,
            MODELS[model_name],
            latencies
        )
        
        all_results[variant_name] = results
        
        print(f"\n   Results:")
        print(f"   Explicit F1: {results['metrics']['overall_explicit_f1']:.3f}")
        print(f"   Implicit F1: {results['metrics']['overall_implicit_f1']:.3f}")
        print(f"   Cost: ${results['estimated_cost']:.4f}")
        print(f"   Avg Latency: {results['avg_latency']*1000:.2f}ms")
    
    # Also run baselines for reference
    print(f"\n{'='*80}")
    print("Running Baselines (for reference)")
    print('='*80)
    
    # Regex
    print("\nRegex...")
    regex_predictions = []
    regex_latencies = []
    for i, sample in enumerate(test_data):
        start_time = time.time()
        pred = extract_entities_regex(sample['text'])
        regex_latencies.append(time.time() - start_time)
        regex_predictions.append(pred)
        print(f"\r   Processed {i + 1}/{len(test_data)} samples...", end='', flush=True)
    print()
    
    regex_results = evaluator.evaluate_model('Regex', regex_predictions,
                                             {'cost_per_1k_input': 0, 'cost_per_1k_output': 0},
                                             regex_latencies)
    all_results['Regex'] = regex_results
    
    # spaCy
    print("\nspaCy...")
    spacy_predictions = []
    spacy_latencies = []
    for i, sample in enumerate(test_data):
        start_time = time.time()
        pred = extract_entities_spacy(sample['text'])
        spacy_latencies.append(time.time() - start_time)
        spacy_predictions.append(pred)
        print(f"\r   Processed {i + 1}/{len(test_data)} samples...", end='', flush=True)
    print()
    
    spacy_results = evaluator.evaluate_model('spaCy', spacy_predictions,
                                             {'cost_per_1k_input': 0, 'cost_per_1k_output': 0},
                                             spacy_latencies)
    all_results['spaCy'] = spacy_results
    
    # Save results
    print(f"\n{'='*80}")
    print("Saving results...")
    print('='*80)
    
    output_dir = Path(__file__).parent.parent / 'outputs'
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = output_dir / f'dspy_variants_comparison_{timestamp}.json'
    
    output = {
        'experiment': {
            'type': 'dspy_variants_comparison',
            'model': model_name,
            'num_samples': num_samples,
            'timestamp': timestamp
        },
        'results': all_results,
        'test_samples': test_data
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"   Results saved to: {output_path}")
    
    # Print comprehensive summary
    print("\n" + "="*80)
    print("COMPREHENSIVE SUMMARY")
    print("="*80)
    
    print("\n" + "-"*80)
    print("EXPLICIT EXTRACTION (Sentence 1)")
    print("-"*80)
    print(f"{'Model':<25} {'F1':>8} {'Cost':>10} {'Latency':>12}")
    print("-"*80)
    for name, results in all_results.items():
        f1 = results['metrics']['overall_explicit_f1']
        cost = results['estimated_cost']
        latency = results['avg_latency'] * 1000
        print(f"{name:<25} {f1:>8.3f} ${cost:>9.4f} {latency:>11.2f}ms")
    
    print("\n" + "-"*80)
    print("IMPLICIT RESOLUTION (Sentence 2) - KEY METRIC")
    print("-"*80)
    print(f"{'Model':<25} {'F1':>8} {'Change':>15}")
    print("-"*80)
    
    baseline_implicit_f1 = all_results['Baseline (original)']['metrics']['overall_implicit_f1']
    
    for name, results in all_results.items():
        f1 = results['metrics']['overall_implicit_f1']
        delta = f1 - baseline_implicit_f1
        delta_str = f"+{delta:.3f}" if delta >= 0 else f"{delta:.3f}"
        print(f"{name:<25} {f1:>8.3f} {delta_str:>15}")
    
    print("\n" + "-"*80)
    print("IMPLICIT BY ENTITY TYPE")
    print("-"*80)
    
    for entity_type in ['PER', 'ORG', 'LOC', 'MISC']:
        print(f"\n{entity_type}:")
        print(f"{'Model':<25} {'F1':>8}")
        print("-"*40)
        for name, results in all_results.items():
            f1 = results['metrics'].get(f'implicit_{entity_type}_f1', 0)
            print(f"{name:<25} {f1:>8.3f}")
    
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    
    # Calculate improvements
    implicit_f1 = all_results['Implicit-aware']['metrics']['overall_implicit_f1']
    cot_f1 = all_results['+ CoT']['metrics']['overall_implicit_f1']
    fewshot_f1 = all_results['+ Few-Shot']['metrics']['overall_implicit_f1']
    both_f1 = all_results['+ CoT + Few-Shot']['metrics']['overall_implicit_f1']
    
    print(f"\n  Baseline → Implicit-aware: {baseline_implicit_f1:.1%} → {implicit_f1:.1%} ({(implicit_f1-baseline_implicit_f1)*100:+.1f}pp)")
    print(f"  + CoT improvement: {(cot_f1-implicit_f1)*100:+.1f}pp")
    print(f"  + Few-Shot improvement: {(fewshot_f1-implicit_f1)*100:+.1f}pp")
    print(f"  + Both improvement: {(both_f1-implicit_f1)*100:+.1f}pp")
    print(f"\n  Best performing: {max(all_results.items(), key=lambda x: x[1]['metrics']['overall_implicit_f1'])[0]}")
    print(f"  Best implicit F1: {both_f1:.1%}")
    
    print("\n" + "="*80)
    
    return output


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Compare DSPy variants for implicit NER')
    parser.add_argument('--model', default='gpt-4o-mini',
                       choices=['gpt-4o-mini', 'gpt-4o', 'o1-mini'],
                       help='Model to use for DSPy')
    parser.add_argument('--samples', type=int, default=100,
                       help='Number of samples to evaluate')
    args = parser.parse_args()
    
    run_experiment(args.model, args.samples)
