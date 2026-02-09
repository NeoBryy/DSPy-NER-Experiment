"""
DSPy Named Entity Recognition - Streamlit Dashboard

Refactored modular architecture for maintainability and testability.
Main orchestrator that coordinates sidebar, experiments, and visualizations.
"""

import streamlit as st
import sys
import asyncio

# Add src to path
sys.path.append('.')

# Import refactored modules
from streamlit_app.components.sidebar import render_sidebar
from streamlit_app.utils.data_loader import load_test_data
from streamlit_app.utils.async_experiment_runner import (
    run_regex_baseline,
    run_spacy_baseline,
    run_dspy_model_async
)

# Page config
st.set_page_config(
    page_title="DSPy NER Demo",
    page_icon="🔬",
    layout="wide"
)

# Title and description
st.title("🔬 DSPy Named Entity Recognition Experiment")

# Intro section with metric definitions
with st.expander("📚 What is Named Entity Recognition (NER)?", expanded=False):
    st.markdown("""
    **Named Entity Recognition** extracts structured information from unstructured text by identifying and classifying entities into predefined categories.
    
    ### Entity Types
    - **PER** (Person): Names of people (e.g., "Tim Cook", "Elon Musk")
    - **ORG** (Organisation): Companies, institutions (e.g., "Apple Inc.", "MIT")
    - **LOC** (Location): Cities, countries, regions (e.g., "San Francisco", "Europe")
    - **MISC** (Miscellaneous): Products, events, other entities (e.g., "iPhone", "Olympics")
    
    ### Example: Explicit vs Implicit Entities
    
    **Explicit Sentence** (straightforward):
    > *"Apple CEO Tim Cook announced the launch of the new iPhone in Cupertino."*
    
    ✅ **Entities to extract:**
    - PER: Tim Cook
    - ORG: Apple
    - LOC: Cupertino
    - MISC: iPhone
    
    **Implicit Sentence** (requires context):
    > *"Tim Cook is CEO of Apple. He announced new products yesterday. The company celebrated record sales."*
    
    ✅ **Explicit entities** (Sentence 1): Tim Cook, Apple  
    🎯 **Implicit references** (Sentences 2+3): "He" → Tim Cook, "The company" → Apple
    
    **The Challenge**: Traditional NER tools typically only extract explicitly mentioned entities. LLMs can extract implicit references like pronouns and descriptions.
    
    **📊 Metrics Explained:**
    - **Precision**: % of extracted entities that were correct
    - **Recall**: % of correct entities that were found
    - **F1 Score**: Balance between precision and recall (higher is better)
    """)

# Approach explanations
with st.expander("🔍 Approaches Compared", expanded=False):
    st.markdown("""
    Let's see how each approach handles this example:
    
    > **Input**: *"Tim Cook announced the iPhone. He praised the engineering team."*
    
    ---
    
    ### 1️⃣ Regex (Rule-Based)
    Uses hand-crafted regular expressions and pattern matching.
    **How it works**:
    - Matches capitalised words: "Tim Cook", "iPhone"
    - Looks for patterns like "Inc.", "Corp.", "LLC"
    - Cannot understand that "He" refers to "Tim Cook" (would have to build specific patterns to handle this)
    
    **Performance on example**:
    - ✅ PER: Tim Cook
    - ✅ MISC: iPhone
    - ❌ "He" → Not extracted (just a pronoun to Regex)
    - ❌ "engineering team" → Missed (no clear pattern)
    
    **Pros**: Executes fast, free, deterministic  
    **Cons**: Brittle, requires user defined and maintained regex patterns for every entity type and custom logic for scenarios like implicit mapping.
    
    ---
    
    ### 2️⃣ spaCy (Traditional ML)
    Pre-trained statistical model (`en_core_web_sm`) trained on web text.
    
    **How it works**:
    - Uses neural network trained on millions of examples
    - Recognises common entity patterns from training data
    - Still cannot resolve implicit references
    
    **Performance on example**:
    - ✅ PER: Tim Cook
    - ✅ MISC: iPhone
    - ❌ "He" → Tagged as pronoun, not linked to "Tim Cook"
    - ⚠️ May or may not catch "engineering team"
    
    **Pros**: Good accuracy, fast inference, works offline, more versatile than regex  
    **Cons**: Limited by training data, **cannot resolve "He" → "Tim Cook"**
    
    ---
    
    ### 3️⃣ DSPy (LLM-Powered)
    Uses large language models (GPT-4o-mini or GPT-4o in our case) with structured prompting.
    
    **How it works**:
    - Understands context: "He" refers back to "Tim Cook"
    - Can follow instructions to extract implicit references
    - Uses Chain-of-Thought reasoning when enabled
    
    **Performance on example**:
    - ✅ PER: Tim Cook, **He** ← Can extract implicit reference!
    - ✅ MISC: iPhone
    - ✅ ORG: engineering team (with proper prompting)
    
    **Example DSPy Prompt**:
    ```
   Your input fields are:
    1. `text` (str): Text containing both explicit entities and implicit references
    Your output fields are:
    1. `reasoning` (str): Step-by-step reasoning: 
        1) Identify explicit entities, 
        2) Find pronouns/references like 'He', 'The company', 
        3) List all entities including implicit refs
    2. `people` (str): List of person names (PER), including pronouns like 'He', 'She'. Comma-separated.
    3. `organizations` (str): List of organization names (ORG), including 'The company', 'The organization'. Comma-separated.
    4. `locations` (str): List of location names (LOC), including 'The city', 'The region'. Comma-separated.
    5. `misc` (str): List of miscellaneous entities (MISC), including 'It', 'The event'. Comma-separated.
    All interactions will be structured in the following way, with the appropriate values filled in.

    [[ ## text ## ]]
    {text}

    [[ ## reasoning ## ]]
    {reasoning}

    [[ ## people ## ]]
    {people}

    [[ ## organizations ## ]]
    {organizations}

    [[ ## locations ## ]]
    {locations}

    [[ ## misc ## ]]
    {misc}

    [[ ## completed ## ]]
    In adhering to this structure, your objective is: 
    Extract named entities with reasoning about implicit references.
    ```

    **Example DSPy Output**:
    ```json
    {
      "PER": ["Tim Cook", "He"],
      "ORG": ["engineering team"],
      "LOC": [],
      "MISC": ["iPhone"]
    }
    ```
    
    **Pros**: Best accuracy, **can extract "He" and link to context**, handles ambiguity  
    **Cons**: Costs money (~$0.0027 per 100 samples), requires API access or local LLM
    """)

# ========== SIDEBAR CONFIGURATION ==========
config = render_sidebar()

# ========== RUN EXPERIMENT ==========
if st.sidebar.button("🚀 Run Experiment", type="primary", use_container_width=True):
    
    # Load test data
    with st.spinner("Loading NER test data..."):
        test_samples, evaluator_class, data_info = load_test_data(
            config['use_implicit'],
            config['sample_size']
        )
        evaluator = evaluator_class(test_samples)
    
    st.success(f"Loaded {len(test_samples)} {data_info} samples")
    
    # Create tabs for results
    tab1, tab2, tab3 = st.tabs(["📊 Metrics Comparison", "🔍 Sample Predictions", "🔬 DSPy Internals"])
    
    # Run experiments
    regex_results, regex_predictions, regex_latencies = run_regex_baseline(test_samples, evaluator)
    spacy_results, spacy_predictions, spacy_latencies = run_spacy_baseline(test_samples, evaluator)
    
    # Run DSPy concurrently using async
    dspy_results, dspy_predictions, dspy_latencies, dspy_histories, extractor_name = asyncio.run(
        run_dspy_model_async(
            test_samples,
            evaluator,
            config['model_name'],
            config['model_config'],
            config['use_implicit'],
            config['use_cot'],
            config['use_fewshot']
        )
    )
    
    # ========== IMPORT DISPLAY COMPONENTS (lazy import to avoid circular dependencies) ==========
    from streamlit_app.components import metrics_display, sample_viewer, dspy_internals
    
    # Display results in tabs
    with tab1:
        metrics_display.render(
            regex_results,
            spacy_results,
            dspy_results,
            extractor_name,
            config['use_implicit']
        )
    
    with tab2:
        sample_viewer.render(
            test_samples,
            regex_predictions,
            spacy_predictions,
            dspy_predictions,
            extractor_name,
            config['use_implicit']
        )
    
    with tab3:
        dspy_internals.render(dspy_histories)
