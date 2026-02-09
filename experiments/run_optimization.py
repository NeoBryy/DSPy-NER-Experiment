"""
Run Automatic Prompt Optimization for Implicit NER.
Uses dspy.BootstrapFewShot to compile the NERExtractorImplicit module.
Compares:
1. Zero-Shot Baseline (Uncompiled)
2. Manual Few-Shot (Human Crafted)
3. Auto-Optimized (DSPy Compiled)
"""

import json
import time
import random
from pathlib import Path
import dspy
from dspy.teleprompt import BootstrapFewShot

from src.config import get_lm, MODELS
from src.modules.entity_extractor_implicit import (
    NERExtractorImplicit,
    NERExtractorCoTFewShot,
    parse_entities,
    ImplicitNERSignature
)
from evaluation.multi_sentence_metrics import (
    MultiSentenceNEREvaluator,
    score_predictions_vs_truth,
    filter_preds_by_text
)


def load_and_split_data(seed=42):
    """Load data and split into Train/Dev/Test."""
    data_path = Path(__file__).parent.parent / 'src' / 'data' / 'ner_multi_sentence_samples.json'
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Shuffle predictably
    random.seed(seed)
    random.shuffle(data)
    
    # Split: 50 Train, 50 Dev, 100 Test
    train_data = data[:50]
    dev_data = data[50:100]
    test_data = data[100:200]
    
    print(f"Data Split: Train={len(train_data)}, Dev={len(dev_data)}, Test={len(test_data)}")
    return train_data, dev_data, test_data


def dspy_metric(gold, pred, trace=None):
    """
    DSPy metric function for optimization.
    Returns weighted F1 score (emphasizing implicit resolution).
    """
    # Parse predictions from the DSPy Prediction object
    # The 'pred' object here is the output of the module (which returns a dict from forward)
    # BUT BootstrapFewShot wraps the module, so we need to handle the output format carefully.
    
    # Check if pred is a dspy.Prediction or dict
    if isinstance(pred, dspy.Prediction):
        # If the module returns a dict, DSPy might wrap it? 
        # Actually our forward() returns a dict, so pred is likely the dict directly?
        # Let's verify standard DSPy behavior:
        # If forward returns dict, optimization might be tricky.
        # Best practice: Output dspy.Prediction in forward for optimization?
        # Our current modules return dicts.
        # Let's adjust to handle dict output.
        pass

    # Actually, for optimization, it's best if the module returns a dspy.Prediction
    # Let's create a wrapper module for optimization that returns Prediction
    return 0.0 # Placeholder, logic below



class TrainableNER(dspy.Module):
    """
    Trainable version of NER extractor.
    Returns dspy.Prediction with raw strings (matching signature) for BootstrapFewShot compatibility.
    """
    def __init__(self):
        super().__init__()
        self.extract = dspy.Predict(ImplicitNERSignature)
    
    def forward(self, text):
        return self.extract(text=text)

def validate_ner(example, pred, trace=None):
    """
    Validation function for BootstrapFewShot.
    Parses raw prediction strings and calculates F1.
    """
    # pred is dspy.Prediction with people, organizations, etc. (strings)
    
    # Parse strings into lists for scoring
    # We can reuse the logic from parse_entities but applied to the Prediction object attributes
    pred_dict = {
        'PER': [e.strip() for e in pred.people.split(',') if e.strip()] if hasattr(pred, 'people') else [],
        'ORG': [e.strip() for e in pred.organizations.split(',') if e.strip()] if hasattr(pred, 'organizations') else [],
        'LOC': [e.strip() for e in pred.locations.split(',') if e.strip()] if hasattr(pred, 'locations') else [],
        'MISC': [e.strip() for e in pred.misc.split(',') if e.strip()] if hasattr(pred, 'misc') else []
    }
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for entity_type in ['PER', 'ORG', 'LOC', 'MISC']:
        preds = set(pred_dict.get(entity_type, []))
        
        # Construct gold set from example
        # note: example fields match signature (people, organizations...) -> strings
        # We need to parse them too!
        gold_str = getattr(example, {'PER': 'people', 'ORG': 'organizations', 'LOC': 'locations', 'MISC': 'misc'}[entity_type], "")
        gold = set([e.strip() for e in gold_str.split(',') if e.strip()])
        
        tp, _ = score_predictions_vs_truth(preds, gold)
        
        total_tp += tp
        total_fp += len(preds) - tp
        total_fn += len(gold) - tp
        
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return f1


def run_optimization(model_name='gpt-4o-mini'):
    print(f"Starting DSPy Prompt Optimization with {model_name}...")
    
    # 1. Load Data
    train_dicts, dev_dicts, test_dicts = load_and_split_data()
    
    # Convert to dspy.Examples which match the ImplicitNERSignature
    # The JSON data has 'sentence1_entities' etc, but our signature expects 'people', 'organizations'...
    # We must transform the data to match the Signature Input/Output for training!
    
    def transform_to_example(d):
        # We need to flatten the "gold" entities into comma-separated strings for 'people', 'organizations' etc.
        # to match the dspy.OutputField expectations of ImplicitNERSignature.
        
        # Combine explicit and implicit for the "gold" string
        per = d['sentence1_entities'].get('PER', []) + [r['text'] for r in d['implicit_refs'] if r['type'] == 'PER']
        org = d['sentence1_entities'].get('ORG', []) + [r['text'] for r in d['implicit_refs'] if r['type'] == 'ORG']
        loc = d['sentence1_entities'].get('LOC', []) + [r['text'] for r in d['implicit_refs'] if r['type'] == 'LOC']
        misc = d['sentence1_entities'].get('MISC', []) + [r['text'] for r in d['implicit_refs'] if r['type'] == 'MISC']
        
        return dspy.Example(
            text=d['text'],
            people=", ".join(per),
            organizations=", ".join(org),
            locations=", ".join(loc),
            misc=", ".join(misc)
        ).with_inputs('text')

    trainset = [transform_to_example(d) for d in train_dicts]
    # We don't necessarily need devset for BootstrapFewShot (it uses trainset to bootstrap), 
    # but good to have if we switched to MIPRO.
    
    # 2. Setup DSPy
    lm = get_lm(model_name)
    dspy.settings.configure(lm=lm)
    
    # 3. Compile (Optimized)
    print("\nCompiling (Optimizing) Model...")
    
    teleprompter = BootstrapFewShot(metric=validate_ner, max_bootstrapped_demos=4, max_labeled_demos=4)
    
    # Use trainable module
    uncompiled_model = TrainableNER()
    
    start_time = time.time()
    compiled_ner = teleprompter.compile(uncompiled_model, trainset=trainset)
    optimize_time = time.time() - start_time
    print(f"Optimization finished in {optimize_time:.2f}s")
    
    # 4. Evaluate Approaches on TEST set
    evaluator = MultiSentenceNEREvaluator(test_dicts)
    
    # Define models to compare
    manual_few_shot_model = NERExtractorCoTFewShot()
    
    approaches = [
        ("Zero-Shot (baseline)", TrainableNER()), 
        ("Manual Few-Shot", manual_few_shot_model), 
        ("Auto-Optimized", compiled_ner)
    ]
    
    all_results = {}
    
    for name, module in approaches:
        print(f"\nEvaluating {name}...")
        predictions = []
        latencies = []
        
        for i, sample in enumerate(test_dicts):
            t0 = time.time()
            text = sample['text']
            
            try:
                # Prediction
                if name == "Manual Few-Shot":
                    # Returns dict directly
                    pred_dict = module(text)
                else:
                    # Returns dspy.Prediction (from TrainableNER)
                    res = module(text)
                    # Convert to dict for evaluator
                    pred_dict = {
                        'PER': [e.strip() for e in res.people.split(',') if e.strip()] if hasattr(res, 'people') else [],
                        'ORG': [e.strip() for e in res.organizations.split(',') if e.strip()] if hasattr(res, 'organizations') else [],
                        'LOC': [e.strip() for e in res.locations.split(',') if e.strip()] if hasattr(res, 'locations') else [],
                        'MISC': [e.strip() for e in res.misc.split(',') if e.strip()] if hasattr(res, 'misc') else []
                    }
            
            except Exception as e:
                print(f"Error predicting sample {i}: {e}")
                pred_dict = {}
                
            latencies.append(time.time() - t0)
            predictions.append(pred_dict)
            
            if (i+1) % 10 == 0:
                print(f"\r   Processed {i+1}/{len(test_dicts)} samples...", end="", flush=True)
        print()
            
        results = evaluator.evaluate_model(name, predictions, MODELS[model_name], latencies)
        all_results[name] = results
        
        print(f"   Implicit F1: {results['metrics']['overall_implicit_f1']:.1%}")
        print(f"   Explicit F1: {results['metrics']['overall_explicit_f1']:.1%}")

    # 5. Save Results
    output_path = Path(__file__).parent.parent / 'outputs' / f'optimization_results_{int(time.time())}.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    run_optimization()
