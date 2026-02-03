"""
Named Entity Recognition using DSPy.
Extracts people, organizations, locations, and miscellaneous entities from text.
"""

import dspy


class EntityExtractor(dspy.Signature):
    """Extract named entities from text. Identify all people (PER), organizations (ORG), locations (LOC), and miscellaneous entities (MISC)."""
    
    text = dspy.InputField(desc="Input text to extract entities from")
    people = dspy.OutputField(desc="List of person names (PER), comma-separated")
    organizations = dspy.OutputField(desc="List of organization names (ORG), comma-separated")
    locations = dspy.OutputField(desc="List of location names (LOC), comma-separated")
    misc = dspy.OutputField(desc="List of miscellaneous entities (MISC), comma-separated")


class NERExtractor(dspy.Module):
    """Named Entity Recognition extractor using DSPy."""
    
    def __init__(self):
        super().__init__()
        self.extract = dspy.ChainOfThought(EntityExtractor)
    
    def forward(self, text):
        """Extract entities from text."""
        result = self.extract(text=text)
        
        # Parse comma-separated lists
        entities = {
            'PER': [e.strip() for e in result.people.split(',') if e.strip()],
            'ORG': [e.strip() for e in result.organizations.split(',') if e.strip()],
            'LOC': [e.strip() for e in result.locations.split(',') if e.strip()],
            'MISC': [e.strip() for e in result.misc.split(',') if e.strip()]
        }
        
        return entities
