"""
Generate multi-sentence NER dataset with explicit then implicit references.
Sentence 1: Explicit entity mentions (e.g., "Tim Cook announced...")
Sentence 2: Implicit references (e.g., "He said it will...")
"""

import json
from pathlib import Path
import random


# Entity data
PEOPLE = [
    {"name": "Tim Cook", "pronoun": "he"},
    {"name": "Elon Musk", "pronoun": "he"},
    {"name": "Bill Gates", "pronoun": "he"},
    {"name": "Jeff Bezos", "pronoun": "he"},
    {"name": "Mark Zuckerberg", "pronoun": "he"},
    {"name": "Sundar Pichai", "pronoun": "he"},
    {"name": "Satya Nadella", "pronoun": "he"},
    {"name": "Susan Wojcicki", "pronoun": "she"},
    {"name": "Sheryl Sandberg", "pronoun": "she"},
]

ORGANIZATIONS = [
    "Apple Inc.", "Microsoft Corporation", "Google LLC", "Amazon", "Meta Platforms",
    "Tesla", "SpaceX", "OpenAI", "Anthropic", "DeepMind",
]

LOCATIONS = [
    "San Francisco", "New York", "London", "Tokyo", "Paris",
    "Silicon Valley", "Seattle", "Boston", "Austin", "Los Angeles",
]

PRODUCTS = ["iPhone 15", "ChatGPT", "Model 3", "Windows 11", "Android"]
EVENTS = ["Olympics", "Super Bowl", "Grammy Awards", "Oscars", "Nobel Prize"]


# Multi-sentence templates: (explicit_sentence, implicit_sentence)
TEMPLATES = [
    # Person + pronoun
    (
        "{PER} announced that {ORG} will launch {MISC}.",
        "{PRONOUN_PER} made the announcement at the conference."
    ),
    (
        "{PER} joined {ORG} as the new CEO.",
        "{PRONOUN_PER} will start next month."
    ),
    (
        "{PER} from {ORG} received the {MISC} award.",
        "{PRONOUN_PER} thanked the team in {LOC}."
    ),
    
    # Organization + "the company"
    (
        "{ORG} is investing heavily in {MISC} technology.",
        "The company plans to expand to {LOC}."
    ),
    (
        "{ORG} announced record profits this quarter.",
        "The company will open new offices in {LOC}."
    ),
    (
        "{PER} said {ORG} is hiring aggressively.",
        "The organization expects growth in {LOC}."
    ),
    
    # Product + "it"
    (
        "{ORG} developed {PRODUCT} in {LOC}.",
        "It will be released next month."
    ),
    (
        "{PER} unveiled {PRODUCT} at the event.",
        "It features revolutionary technology."
    ),
    
    # Event + "the event/ceremony"
    (
        "{LOC} will host the {EVENT} next year.",
        "The event is expected to attract millions."
    ),
    (
        "{PER} attended the {EVENT} in {LOC}.",
        "The ceremony was broadcast globally."
    ),
    
    # Location + "the city/region"
    (
        "{ORG} is opening a new headquarters in {LOC}.",
        "The city will provide tax incentives."
    ),
    (
        "{PER} visited {LOC} to meet with officials.",
        "The region is emerging as a tech hub."
    ),
]


def generate_sample():
    """Generate a multi-sentence NER sample."""
    explicit_template, implicit_template = random.choice(TEMPLATES)
    
    # Select entities
    person = random.choice(PEOPLE) if "{PER}" in explicit_template else None
    org = random.choice(ORGANIZATIONS) if "{ORG}" in explicit_template else None
    loc = random.choice(LOCATIONS) if "{LOC}" in explicit_template else None
    
    # Handle MISC (products vs events)
    misc = None
    if "{PRODUCT}" in explicit_template:
        misc = random.choice(PRODUCTS)
        misc_type = "product"
    elif "{EVENT}" in explicit_template:
        misc = random.choice(EVENTS)
        misc_type = "event"
    elif "{MISC}" in explicit_template:
        misc = random.choice(PRODUCTS + EVENTS)
        misc_type = "product" if misc in PRODUCTS else "event"
    else:
        misc_type = None
    
    # Build explicit sentence
    sent1 = explicit_template
    sent1_entities = {'PER': [], 'ORG': [], 'LOC': [], 'MISC': []}
    
    if person:
        sent1 = sent1.replace('{PER}', person['name'])
        sent1_entities['PER'].append(person['name'])
    if org:
        sent1 = sent1.replace('{ORG}', org)
        sent1_entities['ORG'].append(org)
    if loc:
        sent1 = sent1.replace('{LOC}', loc)
        sent1_entities['LOC'].append(loc)
    if misc:
        sent1 = sent1.replace('{PRODUCT}' if misc_type == 'product' else '{EVENT}', misc)
        sent1 = sent1.replace('{MISC}', misc)
        sent1_entities['MISC'].append(misc)
    
    # Build implicit sentence
    sent2 = implicit_template
    sent2_entities = {'PER': [], 'ORG': [], 'LOC': [], 'MISC': []}
    implicit_refs = []
    
    # Replace pronoun if needed
    if "{PRONOUN_PER}" in sent2 and person:
        pronoun = person['pronoun'].capitalize() if sent2.startswith("{PRONOUN_PER}") else person['pronoun']
        sent2 = sent2.replace('{PRONOUN_PER}', pronoun)
        sent2_entities['PER'].append(person['name'])  # Ground truth: pronoun refers to this person
        implicit_refs.append({
            'text': pronoun,
            'type': 'PER',
            'refers_to': person['name'],
            'implicit_type': 'pronoun'
        })
    
    # Handle "the company/organization"
    if "The company" in sent2 and org:
        sent2_entities['ORG'].append(org)
        implicit_refs.append({
            'text': 'The company',
            'type': 'ORG',
            'refers_to': org,
            'implicit_type': 'the_company'
        })
    elif "The organization" in sent2 and org:
        sent2_entities['ORG'].append(org)
        implicit_refs.append({
            'text': 'The organization',
            'type': 'ORG',
            'refers_to': org,
            'implicit_type': 'the_organization'
        })
    
    # Handle "it" for products
    if sent2.startswith("It ") and misc and misc_type == 'product':
        sent2_entities['MISC'].append(misc)
        implicit_refs.append({
            'text': 'It',
            'type': 'MISC',
            'refers_to': misc,
            'implicit_type': 'pronoun'
        })
    
    # Handle "the event/ceremony"
    if "The event" in sent2 and misc and misc_type == 'event':
        sent2_entities['MISC'].append(misc)
        implicit_refs.append({
            'text': 'The event',
            'type': 'MISC',
            'refers_to': misc,
            'implicit_type': 'the_event'
        })
    elif "The ceremony" in sent2 and misc and misc_type == 'event':
        sent2_entities['MISC'].append(misc)
        implicit_refs.append({
            'text': 'The ceremony',
            'type': 'MISC',
            'refers_to': misc,
            'implicit_type': 'the_ceremony'
        })
    
    # Handle "the city/region"
    if "The city" in sent2 and loc:
        sent2_entities['LOC'].append(loc)
        implicit_refs.append({
            'text': 'The city',
            'type': 'LOC',
            'refers_to': loc,
            'implicit_type': 'the_city'
        })
    elif "The region" in sent2 and loc:
        sent2_entities['LOC'].append(loc)
        implicit_refs.append({
            'text': 'The region',
            'type': 'LOC',
            'refers_to': loc,
            'implicit_type': 'the_region'
        })
    
    # Add any explicit entities in sentence 2 (e.g., new locations)
    if '{LOC}' in implicit_template:
        new_loc = random.choice([l for l in LOCATIONS if l != loc])
        sent2 = sent2.replace('{LOC}', new_loc)
        sent2_entities['LOC'].append(new_loc)
    
    # Combine sentences
    full_text = f"{sent1} {sent2}"
    
    return {
        'text': full_text,
        'sentence1': sent1,
        'sentence2': sent2,
        'sentence1_entities': sent1_entities,
        'sentence2_entities': sent2_entities,
        'implicit_refs': implicit_refs
    }


def generate_dataset(num_records=200):
    """Generate multi-sentence NER dataset."""
    print(f"Generating {num_records} multi-sentence NER samples...")
    print("Sentence 1: Explicit entities")
    print("Sentence 2: Implicit references\n")
    
    samples = []
    for i in range(num_records):
        sample = generate_sample()
        samples.append(sample)
        
        if (i + 1) % 50 == 0:
            print(f"Generated {i + 1}/{num_records} samples...")
    
    # Save
    output_path = Path(__file__).parent.parent / 'src' / 'data' / 'ner_multi_sentence_samples.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved {len(samples)} samples to {output_path}")
    
    # Statistics
    sent1_entities = sum(len(ent) for s in samples for ent in s['sentence1_entities'].values())
    sent2_entities = sum(len(ent) for s in samples for ent in s['sentence2_entities'].values())
    implicit_count = sum(len(s['implicit_refs']) for s in samples)
    
    print(f"\nStatistics:")
    print(f"  Sentence 1 (explicit) entities: {sent1_entities}")
    print(f"  Sentence 2 (ground truth) entities: {sent2_entities}")
    print(f"  Implicit references: {implicit_count}")
    
    # Sample output
    print(f"\nSample entries:")
    for i in range(min(3, len(samples))):
        print(f"\n{i+1}. Full text: {samples[i]['text']}")
        print(f"   Sentence 1: {samples[i]['sentence1']}")
        print(f"   Sentence 1 entities: {samples[i]['sentence1_entities']}")
        print(f"   Sentence 2: {samples[i]['sentence2']}")
        print(f"   Sentence 2 entities: {samples[i]['sentence2_entities']}")
        print(f"   Implicit refs: {samples[i]['implicit_refs']}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate multi-sentence NER data')
    parser.add_argument('--records', type=int, default=200, help='Number of records to generate')
    args = parser.parse_args()
    
    random.seed(42)
    generate_dataset(args.records)
