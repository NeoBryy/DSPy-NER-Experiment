# DSPy Named Entity Recognition Demo

A comprehensive comparison of NER approaches from rule-based to LLM-powered extraction, on examples of sentences with explicit and implicit entities.

## 🎯 What This Demonstrates

This project showcases two major capabilities:

### 1. **Standard NER Comparison**
Evolution from simple pattern matching to modern LLM-powered extraction on explicitly mentioned entities:
- **Regex**: Hand-crafted rules and patterns
- **spaCy**: Pre-trained statistical ML model  
- **DSPy**: Large language model with contextual understanding

### 2. **Implicit Entity Resolution**
Extracting entities that are **not explicitly mentioned** (pronouns, generic references):
- Example: "*Microsoft opened in Seattle. **The city** provided incentives.*"
- Standard NER: Extracts "Microsoft", "Seattle" ✅
- **Implicit NER**: Also extracts "**The city -> Seattle**" as a location entity ✅

## 🔧 How Each Approach Works

### 1️⃣ Regex (Rule-Based)
Uses hand-crafted pattern matching rules:
- **Person Names**: Detects titles (Mr., Dr., President) + capitalised names
- **Organisations**: Matches company suffixes (Inc., Corp., LLC) and acronyms
- **Locations**: Identifies prepositions ("in Paris") and common place names
- **Miscellaneous**: Pattern matches for products, events, and awards

**Pros**: Fast, free, deterministic  
**Cons**: Brittle, requires manual pattern engineering, poor with edge cases, unable to perform implicit entity resolution

### 2️⃣ spaCy (Traditional ML)
Uses a pre-trained statistical model (`en_core_web_sm`):
- **Model**: Trained on OntoNotes 5.0 corpus (news, web, conversation)
- **Architecture**: CNN-based neural network with word embeddings
- **Training**: Supervised learning on millions of annotated examples
- **Entity Mapping**: Maps spaCy's labels (PERSON, GPE, ORG) to our schema

**Pros**: Good accuracy, fast inference, works offline  
**Cons**: Fixed to training data, struggles with domain-specific entities, unable to perform implicit entity resolution

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

### 0. Requirements
  - [Astral UV](https://docs.astral.sh/uv/)
  - [Local LLM Server](https://lmstudio.ai/)(Optional)

### 1. Setup

```bash
# Clone and navigate to project
git clone https://github.com/NeoBryy/DSPy-NER-Experiment.git
cd DSPy-NER-Experiment

# Create virtual environment & install dependencies
uv sync 

# Download spaCy model
uv run spacy download en_core_web_sm

# Generate NER dataset (200 records)
uv run scripts\generate_ner_data.py --records 200 # feel free to edit the examples for each entity!

# Generate Implicit NER dataset (essential for implicit experiments)
uv run scripts\generate_multi_sentence_ner_data.py

# Create .env file with your OpenAI API key https://platform.openai.com/api-keys
echo "OPENAI_API_KEY=your-key-here" > .env
```

**Available Model Online:**
- `gpt-4o-mini` (Input: $0.15 per 1M tokens, Output: $0.60 per 1M tokens)
- `gpt-4o` (Input: $2.50 per 1M tokens, Output: $10.00 per 1M tokens)

**Adding your own local Models:**  
You can also add and use your own LLM locally instead, bringing the token cost down to just your electricity usage...  
All you need to do is edit the src/config.py file with the models you want to run. It is populated with a few examples.


### 2. Run Experiment

```bash
# Run 3-way comparison of explicit NER with 100 samples (terminal)
uv run experiments\run_baseline_comparison.py --samples 100

# Or use a different model (also in terminal)
uv run experiments\run_baseline_comparison.py --model gpt-4o --samples 50
```

### 3. Launch Dashboard

```bash
# This loads the streamlit app, which allows you to configure and run experiments like the
# example above but with nicely formatted visuals and outputs to explore 📈
uv run streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## 🧪 Experiments

The `experiments/` directory contains scripts to test different aspects of NER performance:

### 1. Standard NER Comparison
`experiments/run_baseline_comparison.py`
- **Tests**: Explicit entity extraction on single sentences.
- **Compares**: Regex vs spaCy vs DSPy (Standard).
- **Data**: `src/data/ner_samples.json` (Must generate in advance)
- **Metrics**: Standard Precision, Recall, F1, Cost, Latency.
- **Use case**: General purpose NER benchmarking.

### 2. Multi-Sentence & Implicit NER
`experiments/run_multi_sentence_comparison.py`
- **Tests**: Ability to resolve implicit references across sentences (e.g. "The company" -> "Apple").
- **Compares**: Regex vs spaCy vs DSPy (CoT + Few-Shot).
- **Data**: `src/data/ner_multi_sentence_samples.json` (Must generate in advance)
- **Metrics**: Separates **Explicit F1** (Sentence 1) from **Implicit F1** (Sentence 2).
- **Use case**: verifying that DSPy can handle context that other models miss.

### 3. DSPy Variants Deep Dive
`experiments/run_dspy_variants_comparison.py`
- **Tests**: Impact of different prompting strategies on implicit resolution.
- **Compares**: 5 DSPy variants:
  1. Baseline (Standard)
  2. Implicit-Aware (Prompted)
  3. + Chain-of-Thought
  4. + Few-Shot
  5. + CoT + Few-Shot
- **Use case**: Understanding which prompting technique contributes most to performance.

### 4. Prompt Caching Verification
`experiments/test_prompt_caching.py`
- **Tests**: OpenAI Prompt Caching functionality.
- **Compares**: Token usage across repeated requests.
- **Metrics**: Raw token counts (cached vs uncached).
- **Use case**: Verifying that caching is active and calculating cost savings.

### 5. Automatic Prompt Optimization
`experiments/run_optimization.py`
- **Tests**: Can DSPy's `BootstrapFewShot` optimizer beat manual prompting?
- **Compares**: 
  1. Zero-Shot Baseline (Uncompiled)
  2. Manual Few-Shot (CoT + Hand-picked examples)
  3. Auto-Optimized (DSPy Compiled)
- **Key Finding**: For complex implicit reasoning, **naive auto-optimization (53.5%)** failed to match **manual CoT (72.9%)**. This validates the need for "human-in-the-loop" design for advanced logic tasks.

## 📊 What Gets Measured (Default Configuration)

### Entity Types
- **PER** (Person): Names of people
- **ORG** (Organisation): Companies, institutions
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
├── streamlit_app/                       # Refactored modular Streamlit app
│   ├── components/                      # UI components
│   │   ├── sidebar.py                   # Configuration sidebar with cost calc
│   │   ├── metrics_display.py           # F1 scores, charts, tables
│   │   ├── sample_viewer.py             # Sample predictions with highlighting
│   │   ├── dspy_internals.py            # LLM prompt/response inspection
│   │   └── __init__.py                  # Component exports
│   └── utils/                           # Business logic utilities
│       ├── data_loader.py               # Dataset loading
│       ├── async_experiment_runner.py   # Streamlit async experiment execution
│       ├── experiment_runner.py         # Synchronous experiment runner (legacy)
│       └── __init__.py                  # Utility exports
├── src/
│   ├── modules/
│   │   ├── entity_extractor.py          # Standard DSPy NER
│   │   └── entity_extractor_implicit.py # Implicit NER with CoT/Few-Shot
│   ├── baselines/
│   │   ├── regex_ner.py                 # Regex baseline
│   │   └── spacy_ner.py                 # spaCy baseline
│   ├── utils/
│   │   └── async_runner.py              # Shared async runner for experiments
│   ├── data/
│   │   ├── ner_samples.json             # Standard NER dataset (generated)
│   │   └── ner_multi_sentence_samples.json # Implicit NER dataset (generated)
│   └── config.py                        # Model configurations and pricing
├── evaluation/
│   ├── metrics.py                       # Standard P/R/F1 calculations
│   └── multi_sentence_metrics.py        # Implicit resolution metrics
├── experiments/
│   ├── run_baseline_comparison.py       # Standard NER comparison
│   ├── run_multi_sentence_comparison.py # Implicit NER comparison
│   ├── run_dspy_variants_comparison.py  # CoT/Few-Shot comparison
│   ├── run_optimization.py              # DSPy auto-optimization test
│   └── test_prompt_caching.py           # Prompt caching verification
├── scripts/
│   ├── generate_ner_data.py             # Standard dataset generator
│   └── generate_multi_sentence_ner_data.py # Implicit dataset generator
├── app.py                               # Streamlit dashboard entry point
├── pyproject.toml                       # Python project details
├── .python-version                      # used by uv to specify environment
├── uv.lock                              # uv generated file to specify environment
└── outputs/                             # Experiment results (gitignored)
```

### Concurrent API Execution

DSPy experiments use **concurrent API calls** for ~5x speedup:

**Implementation**:
- **Async/await** with `asyncio` for non-blocking requests
- **Semaphore-based rate limiting** (max 10 concurrent requests)
- **Exponential backoff** retry logic via `tenacity` library
- **Safe concurrency**: Calculated as `(RPM × avg_duration) / 60`

**Performance**:
- Sequential: 20 samples × 2s = 40s
- Concurrent: 20 samples / 5 workers = 8s (**5x faster**)

**Robustness**:
- Auto-retry on rate limit (429) and server errors (5xx)
- Fail-fast on auth errors (401, 403)
- Maintains LLM history capture for debugging

### Prompt Caching (OpenAI)

OpenAI's prompt caching reduces costs by **~50% on cached tokens** when:

**Requirements**:
- Cacheable prefix (system message + few-shot examples) **≥1024 tokens**
- Identical prefix across requests within 5-10 minute window

**Implementation**:
- Disabled DSPy's internal cache (`lm.cache = False`) to capture OpenAI usage data
- Using 6 few-shot examples in `NERExtractorCoTFewShot` to exceed 1024 token threshold

**Cost Savings Example** (100 samples):
- Without caching: ~3,500 prompt tokens × $0.15/1M = $0.000525
- With caching (77% hit): ~800 uncached + 2,700 cached × $0.075/1M = $0.000322
- **Savings**: 38.5% on input tokens

*Note: The dashboard shows cache hit rate next to DSPy cost when available.*

## 🔬 Expected Results

### Standard NER (Explicit Entities)

With `gpt-4o-mini` on 100 samples:

| Metric | Regex | spaCy | DSPy | 
|--------|-------|-------|------|
| Overall F1 | ~0.43 | ~0.70 | ~0.90 |
| Cost | $0.00 | $0.00 | ~$0.003 |
| Latency | ~0.035ms | ~15ms | ~1.6s |

**Key Insight**: Clear progression from rule-based → ML → LLM approaches, with DSPy achieving the highest accuracy through contextual understanding.

### Implicit NER Results

**Testing implicit entity resolution** (can models extract "He", "The city", "The company"?):

| Approach | Implicit F1 | Improvement | Cost (50 samples) |
|----------|-------------|-------------|-------------------|
| Regex | 0.0% | - | $0.00 |
| spaCy | 0.0% | - | $0.00 |
| **DSPy Baseline** | 0.0% | - | $0.0005 |
| **+ Implicit Prompting** | **53.7%** | +53.7pp | $0.0006 |
| **+ Chain-of-Thought** | **79.1%** | +79.1pp | $0.0006 |
| **+ Few-Shot** | **82.6%** | +82.6pp | $0.0006 |
| **+ CoT + Few-Shot** | **87.5%** 🎉 | +87.5pp | $0.0006 |

**Breakthrough Finding**: Proper prompting enables LLMs to extract implicit entity references that traditional NER models completely miss. The combination of Chain-of-Thought reasoning and Few-Shot examples achieves 87.5% F1 on a task where all other approaches score 0%.

## 💡 Why This Matters

This demo shows the trade-offs between different NER approaches:

1. **Regex**: Good for simple, well-defined patterns (e.g., email addresses)
2. **spaCy**: Great for general-purpose NER with good speed/accuracy balance, can train custom model if needed
3. **DSPy**: Best for complex, context-dependent extraction where accuracy is critical

## 📝 Dataset

**Generated during setup** using data generation scripts:

1. **Standard NER**: `scripts/generate_ner_data.py`
   - Creates `src/data/ner_samples.json`
   - Default: 200 records with explicit entities only
   - Used for standard NER benchmarking

2. **Implicit NER**: `scripts/generate_multi_sentence_ner_data.py`
   - Creates `src/data/ner_multi_sentence_samples.json`
   - Contains multi-sentence examples with implicit references (e.g. "The company")
   - Crucial for testing implicit resolution capabilities

**Sample statistics** (200 records):
- 137 Person entities (PER)
- 200 Organization entities (ORG)
- 162 Location entities (LOC)
- 166 Miscellaneous entities (MISC)

Generated using templates to ensure clean, unambiguous labels.

## 🛠️ Customization

### Use Different Models

```bash
uv run experiments\run_baseline_comparison.py --model gpt-4o
```

### Generate Custom Dataset

Customize the number of records or modify entity types:

```bash
# Generate more records
uv run scripts\generate_ner_data.py --records 500

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
