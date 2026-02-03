"""
Regex-based baseline for Named Entity Recognition.
Uses realistic pattern matching that would work in general use cases.
"""

import re


def extract_entities_regex(text):
    """
    Extract named entities using realistic regex patterns.
    
    This is a simplified baseline that uses common heuristics:
    - Capitalized words for potential entities
    - Common titles and suffixes
    - Basic pattern matching
    
    Note: Intentionally simple to show limitations of rule-based approaches.
    """
    
    entities = {
        'PER': [],
        'ORG': [],
        'LOC': [],
        'MISC': []
    }
    
    # === PERSON PATTERNS ===
    # Common titles followed by names
    title_pattern = r'\b(Mr|Mrs|Ms|Dr|Prof|President|Senator|Governor)\.\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)'
    for match in re.finditer(title_pattern, text):
        entities['PER'].append(match.group(2))
    
    # === ORGANIZATION PATTERNS ===
    # Company suffixes (most reliable pattern)
    org_suffix_pattern = r'\b([A-Z][A-Za-z\s&]+?)\s+(Inc\.|Corp\.|Corporation|LLC|Ltd|Limited|University|Institute)\b'
    for match in re.finditer(org_suffix_pattern, text):
        org_name = match.group(1).strip() + ' ' + match.group(2)
        entities['ORG'].append(org_name)
    
    # === LOCATION PATTERNS ===
    # Prepositions often indicate locations
    loc_prep_pattern = r'\b(in|at|from|to)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b'
    for match in re.finditer(loc_prep_pattern, text):
        location = match.group(2)
        # Avoid capturing person names or orgs
        if location not in entities['PER'] and not any(org in location for org in entities['ORG']):
            entities['LOC'].append(location)
    
    # === MISCELLANEOUS PATTERNS ===
    # Products often have version numbers or are in quotes
    product_pattern = r'\b([A-Z][a-z]+)\s+\d+\b'
    for match in re.finditer(product_pattern, text):
        entities['MISC'].append(match.group(0))
    
    # Remove duplicates while preserving order
    for key in entities:
        seen = set()
        unique = []
        for item in entities[key]:
            item_clean = item.strip()
            if item_clean and item_clean not in seen:
                seen.add(item_clean)
                unique.append(item_clean)
        entities[key] = unique
    
    return entities


if __name__ == '__main__':
    # Test the regex extractor
    test_cases = [
        "Apple Inc. CEO Tim Cook announced new products in Cupertino.",
        "Dr. Jane Smith from Stanford University won the Nobel Prize.",
        "President Biden met with European leaders in Paris.",
        "Elon Musk's Tesla and SpaceX are based in California.",
    ]
    
    for text in test_cases:
        result = extract_entities_regex(text)
        print(f"\nText: {text}")
        print(f"Entities: {result}")
