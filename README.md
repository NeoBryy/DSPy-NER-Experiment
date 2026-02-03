# DSPy Named Entity Recognition Demo

A 3-way comparison of NER approaches: **Regex** (rule-based) vs **spaCy** (traditional ML) vs **DSPy** (LLM-powered).

## 🎯 What This Demonstrates

This project showcases the evolution of Named Entity Recognition from simple pattern matching to modern LLM-powered extraction:

1. **Regex**: Hand-crafted rules and patterns
2. **spaCy**: Pre-trained statistical ML model  
3. **DSPy**: Large language model with contextual understanding

## 🔧 How Each Approach Works

### 1️⃣ Regex (Rule-Based)
Uses hand-crafted pattern matching rules:
- **Person Names**: Detects titles (Mr., Dr., President) + capitalised names
- **Organisations**: Matches company suffixes (Inc., Corp., LLC) and acronyms
- **Locations**: Identifies prepositions ("in Paris") and common place names
- **Miscellaneous**: Pattern matches for products, events, and awards

**Pros**: Fast, free, deterministic  
**Cons**: Brittle, requires manual pattern engineering, poor with edge cases

### 2️⃣ spaCy (Traditional ML)
Uses a pre-trained statistical model (`en_core_web_sm`):
- **Model**: Trained on OntoNotes 5.0 corpus (news, web, conversation)
- **Architecture**: CNN-based neural network with word embeddings
- **Training**: Supervised learning on millions of annotated examples
- **Entity Mapping**: Maps spaCy's labels (PERSON, GPE, ORG) to our schema

**Pros**: Good accuracy, fast inference, works offline  
**Cons**: Fixed to training data, struggles with domain-specific entities

### 3️⃣ DSPy (LLM-Powered)
Uses large language models with structured prompting:
- **Prompting**: DSPy generates optimised prompts for entity extraction
- **Signature**: Defines input (text) → output (entities by type) mapping
- **Context**: LLM understands semantic meaning and context
- **Flexibility**: Can extract any entity type without retraining

**Example Prompt**:
```
Extract named entities from the following text.
Classify each entity as PER, ORG, LOC, or MISC.

Text: "Apple CEO Tim Cook announced new products in Cupertino."

Output:
PER: Tim Cook
ORG: Apple
LOC: Cupertino
MISC: None
```

**Pros**: Best accuracy, handles context and ambiguity, no training needed  
**Cons**: Costs money, slower, requires API access or local LLM

## 🚀 Quick Start

### 1. Setup

```powershell
# Clone and navigate to project
cd dspy-llm

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Generate NER dataset (200 records)
python scripts\generate_ner_data.py --records 200

# Create .env file with your OpenAI API key (following .env.example structure)
echo "OPENAI_API_KEY=your-key-here" > .env
```

### 2. Run Experiment

```powershell
# Run 3-way comparison with 100 samples
python experiments\run_baseline_comparison.py --samples 100

# Or use a different model
python experiments\run_baseline_comparison.py --model gpt-4o --samples 50
```

### 3. Launch Dashboard

```powershell
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## 📊 What Gets Measured

### Entity Types
- **PER** (Person): Names of people
- **ORG** (Organization): Companies, institutions
- **LOC** (Location): Cities, countries, regions
- **MISC** (Miscellaneous): Products, events, other entities

- **Precision**: % of extracted entities that were correct
- **Recall**: % of correct entities that were found  
- **F1 Score**: Balance between precision and recall (higher is better)
- **Cost**: Estimated API cost (LLM only)
- **Latency**: Average time per extraction

## 📁 Project Structure

```
dspy-llm/
├── src/
│   ├── modules/
│   │   └── entity_extractor.py    # DSPy NER module
│   ├── baselines/
│   │   ├── regex_ner.py            # Regex baseline
│   │   └── spacy_ner.py            # spaCy baseline
│   ├── data/
│   │   └── ner_samples.json        # Test dataset (200 samples)
│   └── config.py                   # Model configurations
├── evaluation/
│   └── metrics.py                  # P/R/F1 calculations
├── experiments/
│   └── run_baseline_comparison.py  # CLI experiment runner
├── scripts/
│   └── generate_ner_data.py        # Dataset generator
├── app.py                          # Streamlit dashboard
└── outputs/                        # Experiment results
```

## 🔬 Expected Results

With `gpt-4o-mini` on 100 samples:

| Metric | Regex | spaCy | DSPy | 
|--------|-------|-------|------|
| Overall F1 | ~0.65 | ~0.80 | ~0.90 |
| Cost | $0.00 | $0.00 | ~$0.02 |
| Latency | ~0.3ms | ~50ms | ~500ms |

**Key Insight**: Clear progression from rule-based → ML → LLM approaches, with DSPy achieving the highest accuracy through contextual understanding.

## 💡 Why This Matters

This demo shows the trade-offs between different NER approaches:

1. **Regex**: Good for simple, well-defined patterns (e.g., email addresses)
2. **spaCy**: Great for general-purpose NER with good speed/accuracy balance
3. **DSPy**: Best for complex, context-dependent extraction where accuracy is critical

## 📝 Dataset

**Generated during setup** using `scripts/generate_ner_data.py`:
- Creates `src/data/ner_samples.json` with synthetic samples
- Default: 200 records with clean ground truth
- Also includes `src/data/ground_truth.json` (CoNLL-2003 test set)

**Sample statistics** (200 records):
- 137 Person entities (PER)
- 200 Organization entities (ORG)
- 162 Location entities (LOC)
- 166 Miscellaneous entities (MISC)

Generated using templates to ensure clean, unambiguous labels.

## 🛠️ Customization

### Use Different Models

```powershell
python experiments\run_baseline_comparison.py --model gpt-4o
```

### Generate Custom Dataset

Customize the number of records or modify entity types:

```powershell
# Generate more records
python scripts\generate_ner_data.py --records 500

# Edit scripts/generate_ner_data.py to customize entity templates
```

### Modify Entity Types

Edit `src/modules/entity_extractor.py` to add custom entity types.

## 📚 Learn More

- [DSPy Documentation](https://dspy-docs.vercel.app/)
- [DSPy GitHub](https://github.com/stanfordnlp/dspy)
- [spaCy NER](https://spacy.io/usage/linguistic-features#named-entities)
- [Named Entity Recognition](https://en.wikipedia.org/wiki/Named-entity_recognition)

## 📄 License

MIT License - feel free to use for your own experiments!
