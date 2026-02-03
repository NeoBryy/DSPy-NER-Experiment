"""
Run 3-way comparison: Regex vs spaCy vs DSPy for Named Entity Recognition.
"""

import json
import time
from pathlib import Path
from datetime import datetime

import dspy
from src.config import get_lm, MODELS
from src.modules.entity_extractor import NERExtractor
from src.baselines.regex_ner import extract_entities_regex
from src.baselines.spacy_ner import extract_entities_spacy
from evaluation.metrics import ModelEvaluator


def load_test_data():
    """Load NER test samples."""
    data_path = Path(__file__).parent.parent / 'src' / 'data' / 'ner_samples.json'
    with open(data_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def run_experiment(model_name='gpt-4o-mini', num_samples=100):
    """
    Run NER experiment comparing Regex vs spaCy vs DSPy.
    
    Args:
        model_name: LLM model to use for DSPy
        num_samples: Number of samples to evaluate
    """
    print("="*80)
    print(f"NER Extraction Experiment: Regex vs spaCy vs DSPy ({model_name})")
    print("="*80)
    
    # Load test data
    print("\n1. Loading test data...")
    test_data = load_test_data()[:num_samples]
    print(f"   Loaded {len(test_data)} samples")
    
    # Count entities
    total_entities = sum(len(entities) for sample in test_data for entities in sample['entities'].values())
    print(f"   Total entities: {total_entities}")
    for entity_type in ['PER', 'ORG', 'LOC', 'MISC']:
        count = sum(len(sample['entities'][entity_type]) for sample in test_data)
        print(f"   {entity_type}: {count}")
    
    # Initialize evaluator
    evaluator = ModelEvaluator(test_data)
    
    # Run Regex baseline
    print("\n2. Running Regex baseline...")
    regex_predictions = []
    regex_latencies = []
    
    for i, sample in enumerate(test_data):
        start_time = time.time()
        pred = extract_entities_regex(sample['text'])
        regex_latencies.append(time.time() - start_time)
        regex_predictions.append(pred)
        
        # Show progress for every sample
        print(f"\r   Processed {i + 1}/{len(test_data)} samples...", end='', flush=True)
    print()  # New line after progress
    
    regex_results = evaluator.evaluate_model(
        'Regex',
        regex_predictions,
        {'cost_per_1k_input': 0, 'cost_per_1k_output': 0},
        regex_latencies
    )
    
    print(f"\n   Regex Results:")
    print(f"   Overall F1: {regex_results['metrics']['overall_f1']:.3f}")
    print(f"   Avg Latency: {regex_results['avg_latency']*1000:.2f}ms")
    
    # Run spaCy baseline
    print(f"\n3. Running spaCy baseline...")
    spacy_predictions = []
    spacy_latencies = []
    
    for i, sample in enumerate(test_data):
        start_time = time.time()
        pred = extract_entities_spacy(sample['text'])
        spacy_latencies.append(time.time() - start_time)
        spacy_predictions.append(pred)
        
        # Show progress for every sample
        print(f"\r   Processed {i + 1}/{len(test_data)} samples...", end='', flush=True)
    print()  # New line after progress
    
    spacy_results = evaluator.evaluate_model(
        'spaCy',
        spacy_predictions,
        {'cost_per_1k_input': 0, 'cost_per_1k_output': 0},
        spacy_latencies
    )
    
    print(f"\n   spaCy Results:")
    print(f"   Overall F1: {spacy_results['metrics']['overall_f1']:.3f}")
    print(f"   Avg Latency: {spacy_results['avg_latency']*1000:.2f}ms")
    
    # Run DSPy
    print(f"\n4. Running DSPy with {model_name}...")
    lm = get_lm(model_name)
    
    dspy_predictions = []
    dspy_latencies = []
    
    for i, sample in enumerate(test_data):
        with dspy.context(lm=lm):
            extractor = NERExtractor()
            start_time = time.time()
            pred = extractor(sample['text'])
            dspy_latencies.append(time.time() - start_time)
            dspy_predictions.append(pred)
        
        # Show progress for every sample
        print(f"\r   Processed {i + 1}/{len(test_data)} samples...", end='', flush=True)
    print()  # New line after progress
    
    dspy_results = evaluator.evaluate_model(
        f'DSPy ({model_name})',
        dspy_predictions,
        MODELS[model_name],
        dspy_latencies
    )
    
    print(f"\n   DSPy Results:")
    print(f"   Overall F1: {dspy_results['metrics']['overall_f1']:.3f}")
    print(f"   Estimated Cost: ${dspy_results['estimated_cost']:.4f}")
    print(f"   Avg Latency: {dspy_results['avg_latency']*1000:.2f}ms")
    
    # Save results
    print("\n5. Saving results...")
    output_dir = Path(__file__).parent.parent / 'outputs'
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = output_dir / f'ner_results_{timestamp}.json'
    
    results = {
        'experiment': {
            'model': model_name,
            'num_samples': num_samples,
            'timestamp': timestamp
        },
        'regex': regex_results,
        'spacy': spacy_results,
        'dspy': dspy_results,
        'test_samples': test_data,
        'regex_predictions': regex_predictions,
        'spacy_predictions': spacy_predictions,
        'dspy_predictions': dspy_predictions
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"   Results saved to: {output_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nMetric Comparison:")
    print(f"  Overall F1:")
    print(f"    Regex:  {regex_results['metrics']['overall_f1']:.3f}")
    print(f"    spaCy:  {spacy_results['metrics']['overall_f1']:.3f}")
    print(f"    DSPy:   {dspy_results['metrics']['overall_f1']:.3f}")
    
    print(f"\n  By Entity Type (F1):")
    for entity_type in ['PER', 'ORG', 'LOC', 'MISC']:
        regex_f1 = regex_results['metrics'][f'{entity_type}_f1']
        spacy_f1 = spacy_results['metrics'][f'{entity_type}_f1']
        dspy_f1 = dspy_results['metrics'][f'{entity_type}_f1']
        print(f"    {entity_type}: Regex={regex_f1:.3f}, spaCy={spacy_f1:.3f}, DSPy={dspy_f1:.3f}")
    
    print(f"\n  Cost & Latency:")
    print(f"    Regex: $0.0000, {regex_results['avg_latency']*1000:.2f}ms avg")
    print(f"    spaCy: $0.0000, {spacy_results['avg_latency']*1000:.2f}ms avg")
    print(f"    DSPy:  ${dspy_results['estimated_cost']:.4f}, {dspy_results['avg_latency']*1000:.2f}ms avg")
    
    print("\n" + "="*80)
    
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run NER 3-way comparison')
    parser.add_argument('--model', default='gpt-4o-mini',
                       choices=['gpt-4o-mini', 'gpt-4o', 'o1-mini'],
                       help='Model to use for DSPy')
    parser.add_argument('--samples', type=int, default=100,
                       help='Number of samples to evaluate')
    args = parser.parse_args()
    
    run_experiment(args.model, args.samples)