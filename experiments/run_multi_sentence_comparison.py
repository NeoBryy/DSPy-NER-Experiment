"""
Run comparison of Regex vs spaCy vs DSPy on multi-sentence NER.
Evaluates and reports explicit vs implicit performance separately.
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
from evaluation.multi_sentence_metrics import MultiSentenceNEREvaluator
from src.modules.entity_extractor_implicit import NERExtractorCoTFewShot


def load_test_data():
    """Load multi-sentence NER test samples."""
    data_path = Path(__file__).parent.parent / 'src' / 'data' / 'ner_multi_sentence_samples.json'
    with open(data_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def run_experiment(model_name='gpt-4o-mini', num_samples=200):
    """
    Run multi-sentence NER experiment.
    Reports explicit (sentence 1) and implicit (sentence 2) performance separately.
    """
    print("="*80)
    print(f"Multi-Sentence NER Experiment: Regex vs spaCy vs DSPy ({model_name})")
    print("="*80)
    print("\nSentence 1: Explicit entity mentions")
    print("Sentence 2: Implicit references (pronouns, 'the company', etc.)")
    print("\nEvaluates:")
    print("  - EXPLICIT: Extraction from sentence 1")
    print("  - IMPLICIT: Resolution of references in sentence 2")
    
    # Load data
    print(f"\n1. Loading test data...")
    test_data = load_test_data()[:num_samples]
    print(f"   Loaded {len(test_data)} samples")
    
    # Statistics
    sent1_count = sum(len(e) for s in test_data for e in s['sentence1_entities'].values())
    sent2_count = sum(len(e) for s in test_data for e in s['sentence2_entities'].values())
    implicit_count = sum(len(s['implicit_refs']) for s in test_data)
    
    print(f"\n   Sentence 1 entities: {sent1_count}")
    print(f"   Sentence 2 entities: {sent2_count}")
    print(f"   Implicit references: {implicit_count}")
    
    # Initialize evaluator
    evaluator = MultiSentenceNEREvaluator(test_data)
    
    # Run Regex
    print(f"\n2. Running Regex baseline...")
    regex_predictions = []
    regex_latencies = []
    
    for i, sample in enumerate(test_data):
        start_time = time.time()
        pred = extract_entities_regex(sample['text'])
        regex_latencies.append(time.time() - start_time)
        regex_predictions.append(pred)
        
        print(f"\r   Processed {i + 1}/{len(test_data)} samples...", end='', flush=True)
    print()
    
    regex_results = evaluator.evaluate_model(
        'Regex',
        regex_predictions,
        {'cost_per_1k_input': 0, 'cost_per_1k_output': 0},
        regex_latencies
    )
    
    print(f"   Explicit F1: {regex_results['metrics']['overall_explicit_f1']:.3f}")
    print(f"   Implicit F1: {regex_results['metrics']['overall_implicit_f1']:.3f}")
    
    # Run spaCy
    print(f"\n3. Running spaCy baseline...")
    spacy_predictions = []
    spacy_latencies = []
    
    for i, sample in enumerate(test_data):
        start_time = time.time()
        pred = extract_entities_spacy(sample['text'])
        spacy_latencies.append(time.time() - start_time)
        spacy_predictions.append(pred)
        
        print(f"\r   Processed {i + 1}/{len(test_data)} samples...", end='', flush=True)
    print()
    
    spacy_results = evaluator.evaluate_model(
        'spaCy',
        spacy_predictions,
        {'cost_per_1k_input': 0, 'cost_per_1k_output': 0},
        spacy_latencies
    )
    
    print(f"   Explicit F1: {spacy_results['metrics']['overall_explicit_f1']:.3f}")
    print(f"   Implicit F1: {spacy_results['metrics']['overall_implicit_f1']:.3f}")
    
    # Run DSPy
    print(f"\n4. Running DSPy with {model_name}...")
    lm = get_lm(model_name)
    
    dspy_predictions = []
    dspy_latencies = []
    
    for i, sample in enumerate(test_data):
        with dspy.context(lm=lm):
            # Use the advanced implicit extractor to test implicit resolution
            extractor = NERExtractorCoTFewShot()
            start_time = time.time()
            pred = extractor(sample['text'])
            dspy_latencies.append(time.time() - start_time)
            dspy_predictions.append(pred)
        
        print(f"\r   Processed {i + 1}/{len(test_data)} samples...", end='', flush=True)
    print()
    
    dspy_results = evaluator.evaluate_model(
        f'DSPy ({model_name})',
        dspy_predictions,
        MODELS[model_name],
        dspy_latencies
    )
    
    print(f"   Explicit F1: {dspy_results['metrics']['overall_explicit_f1']:.3f}")
    print(f"   Implicit F1: {dspy_results['metrics']['overall_implicit_f1']:.3f}")
    
    # Save results
    print(f"\n5. Saving results...")
    output_dir = Path(__file__).parent.parent / 'outputs'
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = output_dir / f'multi_sentence_ner_results_{timestamp}.json'
    
    results = {
        'experiment': {
            'type': 'multi_sentence_ner',
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
    print("SUMMARY - EXPLICIT EXTRACTION (Sentence 1)")
    print("="*80)
    print(f"Overall F1:")
    print(f"    Regex:  {regex_results['metrics']['overall_explicit_f1']:.3f}")
    print(f"    spaCy:  {spacy_results['metrics']['overall_explicit_f1']:.3f}")
    print(f"    DSPy:   {dspy_results['metrics']['overall_explicit_f1']:.3f}")
    
    print(f"\nBy Entity Type (F1):")
    for entity_type in ['PER', 'ORG', 'LOC', 'MISC']:
        regex_f1 = regex_results['metrics'].get(f'explicit_{entity_type}_f1', 0)
        spacy_f1 = spacy_results['metrics'].get(f'explicit_{entity_type}_f1', 0)
        dspy_f1 = dspy_results['metrics'].get(f'explicit_{entity_type}_f1', 0)
        count = int(regex_results['metrics'].get(f'explicit_{entity_type}_count', 0))
        
        if count > 0:
            print(f"    {entity_type} (n={count}): Regex={regex_f1:.3f}, spaCy={spacy_f1:.3f}, DSPy={dspy_f1:.3f}")
    
    print("\n" + "="*80)
    print("SUMMARY - IMPLICIT RESOLUTION (Sentence 2)")
    print("="*80)
    print(f"Overall F1:")
    print(f"    Regex:  {regex_results['metrics']['overall_implicit_f1']:.3f}")
    print(f"    spaCy:  {spacy_results['metrics']['overall_implicit_f1']:.3f}")
    print(f"    DSPy:   {dspy_results['metrics']['overall_implicit_f1']:.3f}")
    
    print(f"\nBy Entity Type (F1):")
    for entity_type in ['PER', 'ORG', 'LOC', 'MISC']:
        regex_f1 = regex_results['metrics'].get(f'implicit_{entity_type}_f1', 0)
        spacy_f1 = spacy_results['metrics'].get(f'implicit_{entity_type}_f1', 0)
        dspy_f1 = dspy_results['metrics'].get(f'implicit_{entity_type}_f1', 0)
        count = int(regex_results['metrics'].get(f'implicit_{entity_type}_count', 0))
        
        if count > 0:
            print(f"    {entity_type} (n={count}): Regex={regex_f1:.3f}, spaCy={spacy_f1:.3f}, DSPy={dspy_f1:.3f}")
    
    print("\n" + "="*80)
    print("COST & LATENCY")
    print("="*80)
    print(f"    Regex: $0.0000, {regex_results['avg_latency']*1000:.2f}ms avg")
    print(f"    spaCy: $0.0000, {spacy_results['avg_latency']*1000:.2f}ms avg")
    print(f"    DSPy:  ${dspy_results['estimated_cost']:.4f}, {dspy_results['avg_latency']*1000:.2f}ms avg")
    print("="*80)
    
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run multi-sentence NER comparison')
    parser.add_argument('--model', default='gpt-4o-mini',
                       choices=['gpt-4o-mini', 'gpt-4o', 'o1-mini'],
                       help='Model to use for DSPy')
    parser.add_argument('--samples', type=int, default=200,
                       help='Number of samples to evaluate')
    args = parser.parse_args()
    
    run_experiment(args.model, args.samples)
