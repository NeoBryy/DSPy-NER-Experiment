"""
Generate synthetic NER dataset for demo.
Creates realistic sentences with known entities for clean ground truth.
"""

import json
from pathlib import Path
import random


# Entity templates
PEOPLE = [
    "Tim Cook", "Elon Musk", "Bill Gates", "Jeff Bezos", "Mark Zuckerberg",
    "Sundar Pichai", "Satya Nadella", "Jack Dorsey", "Susan Wojcicki", "Sheryl Sandberg",
    "Dr. Anthony Fauci", "President Biden", "Senator Warren", "Mayor Adams", "Judge Roberts"
]

ORGANIZATIONS = [
    "Apple Inc.", "Microsoft Corporation", "Google LLC", "Amazon", "Meta Platforms",
    "Tesla", "SpaceX", "OpenAI", "Anthropic", "DeepMind",
    "Stanford University", "MIT", "Harvard", "NASA", "WHO"
]

LOCATIONS = [
    "San Francisco", "New York", "London", "Tokyo", "Paris",
    "Silicon Valley", "Seattle", "Boston", "Austin", "Los Angeles",
    "California", "United States", "Europe", "Asia", "China"
]

MISC = [
    "iPhone 15", "ChatGPT", "Model 3", "Windows 11", "Android",
    "Nobel Prize", "Olympics", "Super Bowl", "Grammy Awards", "Oscars"
]

# Sentence templates
TEMPLATES = [
    "{PER} announced that {ORG} will launch {MISC} in {LOC}.",
    "{ORG} CEO {PER} spoke at a conference in {LOC} about {MISC}.",
    "The {MISC} was developed by {ORG} researchers in {LOC}.",
    "{PER} from {ORG} won the {MISC} award in {LOC}.",
    "{ORG} is opening a new office in {LOC}, according to {PER}.",
    "At {ORG}, {PER} is leading the {MISC} project.",
    "{PER} traveled to {LOC} to meet with {ORG} executives.",
    "The {MISC} event will be held in {LOC}, sponsored by {ORG}.",
    "{ORG} announced a partnership with {ORG2} in {LOC}.",
    "{PER} and {PER2} discussed {MISC} at the {LOC} summit.",
    "Researchers at {ORG} in {LOC} are working on {MISC}.",
    "{PER} joined {ORG} as the new head of {MISC} division.",
    "The {MISC} conference in {LOC} featured speakers from {ORG}.",
    "{ORG} is investing heavily in {MISC} technology in {LOC}.",
    "{PER} criticized {ORG} for their handling of {MISC}.",
]


def generate_sample():
    """Generate a single NER sample."""
    template = random.choice(TEMPLATES)
    
    # Track entities used
    entities = {
        'PER': [],
        'ORG': [],
        'LOC': [],
        'MISC': []
    }
    
    # Fill template
    text = template
    
    # Replace placeholders
    if '{PER}' in text:
        per = random.choice(PEOPLE)
        entities['PER'].append(per)
        text = text.replace('{PER}', per, 1)
    
    if '{PER2}' in text:
        per2 = random.choice([p for p in PEOPLE if p not in entities['PER']])
        entities['PER'].append(per2)
        text = text.replace('{PER2}', per2, 1)
    
    if '{ORG}' in text:
        org = random.choice(ORGANIZATIONS)
        entities['ORG'].append(org)
        text = text.replace('{ORG}', org, 1)
    
    if '{ORG2}' in text:
        org2 = random.choice([o for o in ORGANIZATIONS if o not in entities['ORG']])
        entities['ORG'].append(org2)
        text = text.replace('{ORG2}', org2, 1)
    
    if '{LOC}' in text:
        loc = random.choice(LOCATIONS)
        entities['LOC'].append(loc)
        text = text.replace('{LOC}', loc, 1)
    
    if '{MISC}' in text:
        misc = random.choice(MISC)
        entities['MISC'].append(misc)
        text = text.replace('{MISC}', misc, 1)
    
    return {
        'text': text,
        'entities': entities
    }


def generate_dataset(num_records=200):
    """Generate synthetic NER dataset.
    
    Args:
        num_records: Number of NER samples to generate
    """
    
    print(f"Generating {num_records} synthetic NER samples...")
    
    samples = []
    for i in range(num_records):
        sample = generate_sample()
        samples.append(sample)
        
        if (i + 1) % 50 == 0:
            print(f"Generated {i + 1}/{num_records} samples...")
    
    # Save to data directory
    output_path = Path(__file__).parent.parent / 'src' / 'data' / 'ner_samples.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved {len(samples)} samples to {output_path}")
    
    # Print statistics
    total_entities = sum(len(entities) for sample in samples for entities in sample['entities'].values())
    print(f"\nStatistics:")
    print(f"  Total sentences: {len(samples)}")
    print(f"  Total entities: {total_entities}")
    
    # Count by type
    for entity_type in ['PER', 'ORG', 'LOC', 'MISC']:
        count = sum(len(sample['entities'][entity_type]) for sample in samples)
        print(f"  {entity_type}: {count}")
    
    # Print samples
    print(f"\nSample entries:")
    for i in range(min(3, len(samples))):
        print(f"\n{i+1}. {samples[i]['text']}")
        print(f"   Entities: {samples[i]['entities']}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate synthetic NER data')
    parser.add_argument('--records', type=int, default=200, help='Number of records to generate')
    args = parser.parse_args()
    
    random.seed(42)  # For reproducibility
    generate_dataset(args.records)
