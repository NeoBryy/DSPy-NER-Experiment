"""
Enhanced NER extractors with Chain-of-Thought and Few-Shot for implicit resolution.
"""

import dspy
import json


class ImplicitNERSignature(dspy.Signature):
    """Extract named entities including implicit references like 'He', 'The city', 'The company'."""
    
    text = dspy.InputField(desc="Text containing both explicit entities and implicit references")
    people = dspy.OutputField(desc="List of person names (PER), including pronouns like 'He', 'She'. Comma-separated.")
    organizations = dspy.OutputField(desc="List of organization names (ORG), including references like 'The company', 'The organization'. Comma-separated.")
    locations = dspy.OutputField(desc="List of location names (LOC), including references like 'The city', 'The region'. Comma-separated.")
    misc = dspy.OutputField(desc="List of miscellaneous entities (MISC), including 'It', 'The event'. Comma-separated.")


class ImplicitNERWithCoT(dspy.Signature):
    """Extract named entities with reasoning about implicit references."""
    
    text = dspy.InputField(desc="Text containing both explicit entities and implicit references")
    reasoning = dspy.OutputField(
        desc="Step-by-step reasoning: 1) Identify explicit entities, 2) Find pronouns/references like 'He', 'The company', 3) List all entities including implicit refs"
    )
    people = dspy.OutputField(desc="List of person names (PER), including pronouns like 'He', 'She'. Comma-separated.")
    organizations = dspy.OutputField(desc="List of organization names (ORG), including 'The company', 'The organization'. Comma-separated.")
    locations = dspy.OutputField(desc="List of location names (LOC), including 'The city', 'The region'. Comma-separated.")
    misc = dspy.OutputField(desc="List of miscellaneous entities (MISC), including 'It', 'The event'. Comma-separated.")


def parse_entities(result):
    """Parse DSPy result into entity dict."""
    return {
        'PER': [e.strip() for e in result.people.split(',') if e.strip()],
        'ORG': [e.strip() for e in result.organizations.split(',') if e.strip()],
        'LOC': [e.strip() for e in result.locations.split(',') if e.strip()],
        'MISC': [e.strip() for e in result.misc.split(',') if e.strip()]
    }


class NERExtractorImplicit(dspy.Module):
    """Basic NER extractor told to include implicit references."""
    
    def __init__(self):
        super().__init__()
        self.extract = dspy.Predict(ImplicitNERSignature)
    
    def forward(self, text):
        result = self.extract(text=text)
        return parse_entities(result)


class NERExtractorCoT(dspy.Module):
    """NER extractor with Chain-of-Thought reasoning for implicit references."""
    
    def __init__(self):
        super().__init__()
        self.extract = dspy.ChainOfThought(ImplicitNERWithCoT)
    
    def forward(self, text):
        result = self.extract(text=text)
        return parse_entities(result)


class NERExtractorFewShot(dspy.Module):
    """NER extractor with few-shot examples showing implicit reference extraction."""
    
    def __init__(self):
        super().__init__()
        
        # Few-shot examples demonstrating implicit reference extraction
        self.examples = [
            dspy.Example(
                text="Tim Cook announced new products. He praised the team.",
                people="Tim Cook, He",
                organizations="",
                locations="",
                misc=""
            ).with_inputs('text'),
            dspy.Example(
                text="Apple Inc. reported record profits. The company will expand to Europe.",
                people="",
                organizations="Apple Inc., The company",
                locations="Europe",
                misc=""
            ).with_inputs('text'),
            dspy.Example(
                text="Microsoft opened an office in Seattle. The city provided incentives.",
                people="",
                organizations="Microsoft",
                locations="Seattle, The city",
                misc=""
            ).with_inputs('text'),
        ]
        
        self.extract = dspy.Predict(ImplicitNERSignature)
    
    def forward(self, text):
        # Use demos (few-shot examples) in prediction
        with dspy.context(lm=dspy.settings.lm):
            result = self.extract(text=text, demos=self.examples)
        return parse_entities(result)


class NERExtractorCoTFewShot(dspy.Module):
    """NER extractor with BOTH Chain-of-Thought AND Few-Shot."""
    
    def __init__(self):
        super().__init__()
        
        # Few-shot examples with reasoning (need 5+ examples to exceed 1024 token threshold)
        self.examples = [
            dspy.Example(
                text="Tim Cook announced new products. He praised the team.",
                reasoning="1) Explicit entities: 'Tim Cook' (PER). 2) Implicit references: 'He' is a pronoun referring to a person. 3) Include: Tim Cook, He",
                people="Tim Cook, He",
                organizations="",
                locations="",
                misc=""
            ).with_inputs('text'),
            dspy.Example(
                text="Apple Inc. reported record profits. The company will expand to Europe.",
                reasoning="1) Explicit: 'Apple Inc.' (ORG), 'Europe' (LOC). 2) Implicit: 'The company' refers to an organization. 3) Include: Apple Inc., The company, Europe",
                people="",
                organizations="Apple Inc., The company",
                locations="Europe",
                misc=""
            ).with_inputs('text'),
            dspy.Example(
                text="Microsoft opened an office in Seattle. The city provided incentives.",
                reasoning="1) Explicit: 'Microsoft' (ORG), 'Seattle' (LOC). 2) Implicit: 'The city' refers to a location. 3) Include: Microsoft, Seattle, The city",
                people="",
                organizations="Microsoft",
                locations="Seattle, The city",
                misc=""
            ).with_inputs('text'),
            dspy.Example(
                text="Amazon announced a new initiative in New York. The initiative focuses on sustainability.",
                reasoning="1) Explicit: 'Amazon' (ORG), 'New York' (LOC). 2) Implicit: 'The initiative' refers to a miscellaneous entity. 3) Include: Amazon, New York, The initiative",
                people="",
                organizations="Amazon",
                locations="New York",
                misc="The initiative"
            ).with_inputs('text'),
            dspy.Example(
                text="Jeff Bezos met with government officials in Washington DC. He discussed policy changes.",
                reasoning="1) Explicit: 'Jeff Bezos' (PER), 'Washington DC' (LOC). 2) Implicit: 'He' refers to Jeff Bezos. 3) Include: Jeff Bezos, He, Washington DC",
                people="Jeff Bezos, He",
                organizations="",
                locations="Washington DC",
                misc=""
            ).with_inputs('text'),
            dspy.Example(
                text="Tesla announced its quarterly earnings in Austin, Texas. The company exceeded analyst expectations. It plans to expand manufacturing capacity.",
                reasoning="1) Explicit: 'Tesla' (ORG), 'Austin' (LOC), 'Texas' (LOC). 2) Implicit: 'The company' refers to Tesla, 'It' refers to Tesla. 3) Include: Tesla, The company, It, Austin, Texas",
                people="",
                organizations="Tesla, The company, It",
                locations="Austin, Texas",
                misc=""
            ).with_inputs('text'),
        ]
        
        self.extract = dspy.ChainOfThought(ImplicitNERWithCoT)
    
    def forward(self, text):
        # Use demos with CoT
        with dspy.context(lm=dspy.settings.lm):
            result = self.extract(text=text, demos=self.examples)
        return parse_entities(result)

