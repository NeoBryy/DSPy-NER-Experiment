"""
SpaCy-based baseline for Named Entity Recognition.
Uses spaCy's pre-trained NER model.
"""

import spacy


# Load spaCy model (will be loaded once and reused)
_nlp = None

def get_spacy_model():
    """Load spaCy model (cached)."""
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            # Model not installed, download it
            import subprocess
            import sys
            print("Downloading spaCy model 'en_core_web_sm'...")
            subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
            _nlp = spacy.load("en_core_web_sm")
    return _nlp


def extract_entities_spacy(text):
    """
    Extract named entities using spaCy's pre-trained NER model.
    
    Maps spaCy entity types to our schema:
    - PERSON → PER
    - ORG → ORG
    - GPE, LOC, FAC → LOC
    - PRODUCT, EVENT, WORK_OF_ART, LAW, LANGUAGE → MISC
    """
    
    nlp = get_spacy_model()
    doc = nlp(text)
    
    entities = {
        'PER': [],
        'ORG': [],
        'LOC': [],
        'MISC': []
    }
    
    # Map spaCy labels to our schema
    for ent in doc.ents:
        entity_text = ent.text.strip()
        
        if ent.label_ == 'PERSON':
            entities['PER'].append(entity_text)
        elif ent.label_ == 'ORG':
            entities['ORG'].append(entity_text)
        elif ent.label_ in ['GPE', 'LOC', 'FAC']:  # Geopolitical entity, location, facility
            entities['LOC'].append(entity_text)
        elif ent.label_ in ['PRODUCT', 'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE', 'NORP']:
            entities['MISC'].append(entity_text)
        # Skip: DATE, TIME, PERCENT, MONEY, QUANTITY, ORDINAL, CARDINAL
    
    # Remove duplicates while preserving order
    for key in entities:
        seen = set()
        unique = []
        for item in entities[key]:
            if item and item not in seen:
                seen.add(item)
                unique.append(item)
        entities[key] = unique
    
    return entities


if __name__ == '__main__':
    # Test the spaCy extractor
    test_cases = [
        "Apple Inc. CEO Tim Cook announced new products in Cupertino.",
        "Dr. Jane Smith from Stanford University won the Nobel Prize.",
        "President Biden met with European leaders in Paris.",
        "Elon Musk's Tesla and SpaceX are based in California.",
    ]
    
    for text in test_cases:
        result = extract_entities_spacy(text)
        print(f"\nText: {text}")
        print(f"Entities: {result}")
