import streamlit as st
import json
import sys
import os
import time
from datetime import datetime
import plotly.graph_objects as go
import pandas as pd

# Add src to path
sys.path.append('.')

from src.baselines.regex_ner import extract_entities_regex
from src.baselines.spacy_ner import extract_entities_spacy
from src.modules.entity_extractor import NERExtractor
from src.config import get_lm, MODELS
from evaluation.metrics import ModelEvaluator
import dspy

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
    - **ORG** (Organization): Companies, institutions (e.g., "Apple Inc.", "MIT")
    - **LOC** (Location): Cities, countries, regions (e.g., "San Francisco", "Europe")
    - **MISC** (Miscellaneous): Products, events, other entities (e.g., "iPhone", "Olympics")
    
    **📊 Metrics Explained:**
    - **Precision**: % of extracted entities that were correct
    - **Recall**: % of correct entities that were found
    - **F1 Score**: Balance between precision and recall (higher is better)
    - **Cost**: Estimated API cost (LLM only)
    - **Latency**: Average time per extraction
    """)

with st.expander("🔧 How Each Approach Works", expanded=False):
    st.markdown("""
    ### 1️⃣ Regex (Rule-Based)
    Uses hand-crafted pattern matching rules:
    - **Person Names**: Detects titles (Mr., Dr., President) + capitalized names
    - **Organizations**: Matches Company suffixes (Inc., Corp., LLC) and acronyms
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
    - **Prompting**: DSPy generates optimized prompts for entity extraction
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
    **Cons**: Costs money, slower, requires API access
    """)

# Sidebar configuration
st.sidebar.header("⚙️ Experiment Configuration")

# Model selection
model_name = st.sidebar.selectbox(
    "Select LLM Model",
    options=list(MODELS.keys()),
    index=0,
    help="Choose which OpenAI model to use for DSPy extraction"
)

# Sample size
sample_size = st.sidebar.slider(
    "Records to Test",
    min_value=5,
    max_value=100,
    value=20,
    step=5,
    help="More records = more accurate results but higher cost"
)

# Display model info with estimated cost
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Model Info")
model_config = MODELS[model_name]

# Estimate cost based on typical token usage
# Assume ~100 input tokens and ~50 output tokens per sample
avg_input_tokens = 100 * sample_size
avg_output_tokens = 50 * sample_size
estimated_cost = (
    (avg_input_tokens / 1000) * model_config['cost_per_1k_input'] +
    (avg_output_tokens / 1000) * model_config['cost_per_1k_output']
)

st.sidebar.write(f"**Estimated Cost:** ${estimated_cost:.4f}")
st.sidebar.caption(f"Based on {sample_size} records with {model_name}")


def format_time(seconds):
    """Format time in appropriate units."""
    if seconds < 0.001:
        return f"{seconds * 1000000:.0f}µs"
    elif seconds < 1:
        return f"{seconds * 1000:.2f}ms"
    else:
        return f"{seconds:.2f}s"


# Run experiment button
if st.sidebar.button("🚀 Run Experiment", type="primary", use_container_width=True):
    
    # Load test data
    with st.spinner("Loading NER test data..."):
        with open('src/data/ner_samples.json', 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        test_samples = test_data[:sample_size]
    
    st.success(f"Loaded {len(test_samples)} samples")
    
    # Create tabs for results
    tab1, tab2, tab3 = st.tabs(["� Metrics Comparison", "🔍 Sample Predictions", "🔬 DSPy Internals"])
    
    # ===== REGEX BASELINE =====
    with st.spinner("Running Regex baseline..."):
        regex_predictions = []
        regex_latencies = []
        progress_bar = st.progress(0)
        for i, sample in enumerate(test_samples):
            start_time = time.time()
            pred = extract_entities_regex(sample['text'])
            regex_latencies.append(time.time() - start_time)
            regex_predictions.append(pred)
            progress_bar.progress((i + 1) / len(test_samples))
        progress_bar.empty()
        
        evaluator = ModelEvaluator(test_samples)
        regex_results = evaluator.evaluate_model(
            'Regex Baseline',
            regex_predictions,
            {'cost_per_1k_input': 0, 'cost_per_1k_output': 0},
            latencies=regex_latencies
        )
    
    st.success("✅ Regex baseline complete")
    
    # ===== SPACY BASELINE =====
    with st.spinner("Running spaCy baseline..."):
        spacy_predictions = []
        spacy_latencies = []
        progress_bar = st.progress(0)
        for i, sample in enumerate(test_samples):
            start_time = time.time()
            pred = extract_entities_spacy(sample['text'])
            spacy_latencies.append(time.time() - start_time)
            spacy_predictions.append(pred)
            progress_bar.progress((i + 1) / len(test_samples))
        progress_bar.empty()
        
        spacy_results = evaluator.evaluate_model(
            'spaCy',
            spacy_predictions,
            {'cost_per_1k_input': 0, 'cost_per_1k_output': 0},
            latencies=spacy_latencies
        )
    
    st.success("✅ spaCy baseline complete")
    
    # ===== DSPy MODEL =====
    with st.spinner(f"Running DSPy with {model_name}..."):
        lm = get_lm(model_name)
        dspy_predictions = []
        dspy_latencies = []
        dspy_histories = []  # Store LLM interaction history
        
        progress_bar = st.progress(0)
        progress_text = st.empty()
        for i, sample in enumerate(test_samples):
            progress_text.text(f"Processing sample {i+1}/{len(test_samples)}...")
            with dspy.context(lm=lm):
                extractor = NERExtractor()
                start_time = time.time()
                pred = extractor(sample['text'])
                dspy_latencies.append(time.time() - start_time)
                dspy_predictions.append(pred)
                
                # Capture the actual LLM prompt and response from lm.history
                if hasattr(lm, 'history') and lm.history:
                    last_interaction = lm.history[-1]
                    dspy_histories.append({
                        'text': sample['text'],
                        'prediction': pred,
                        'messages': last_interaction.get('messages', []),
                        'response': last_interaction.get('response', None),
                        'outputs': last_interaction.get('outputs', [])
                    })
                else:
                    dspy_histories.append({
                        'text': sample['text'],
                        'prediction': pred,
                        'messages': [],
                        'response': None,
                        'outputs': []
                    })
            progress_bar.progress((i + 1) / len(test_samples))
        progress_bar.empty()
        progress_text.empty()
        
        dspy_results = evaluator.evaluate_model(
            f'DSPy ({model_name})',
            dspy_predictions,
            model_config,
            latencies=dspy_latencies
        )
    
    st.success("✅ DSPy extraction complete")
    
    # ===== METRICS COMPARISON TAB =====
    with tab1:
        st.header("📊 Performance Comparison")
        
        # Overall F1 as KPI boxes
        st.subheader("Overall F1 Score")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "🔧 Regex",
                f"{regex_results['metrics']['overall_f1']:.1%}",
                help="Rule-based pattern matching"
            )
        
        with col2:
            delta_vs_regex = spacy_results['metrics']['overall_f1'] - regex_results['metrics']['overall_f1']
            st.metric(
                "🧠 spaCy",
                f"{spacy_results['metrics']['overall_f1']:.1%}",
                delta=f"{delta_vs_regex:+.1%} from Regex",
                help="Traditional ML model"
            )
        
        with col3:
            delta_vs_regex = dspy_results['metrics']['overall_f1'] - regex_results['metrics']['overall_f1']
            st.metric(
                "🤖 DSPy",
                f"{dspy_results['metrics']['overall_f1']:.1%}",
                delta=f"{delta_vs_regex:+.1%} from Regex",
                help="LLM-powered extraction"
            )
        
        st.markdown("---")
        
        # F1 by entity type
        st.subheader("F1 Score by Entity Type")
        
        entity_types = ['PER', 'ORG', 'LOC', 'MISC']
        regex_f1s = [regex_results['metrics'][f'{et}_f1'] for et in entity_types]
        spacy_f1s = [spacy_results['metrics'][f'{et}_f1'] for et in entity_types]
        dspy_f1s = [dspy_results['metrics'][f'{et}_f1'] for et in entity_types]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Regex',
            x=entity_types,
            y=regex_f1s,
            marker_color='#FF6B6B'
        ))
        fig.add_trace(go.Bar(
            name='spaCy',
            x=entity_types,
            y=spacy_f1s,
            marker_color='#95E1D3'
        ))
        fig.add_trace(go.Bar(
            name='DSPy',
            x=entity_types,
            y=dspy_f1s,
            marker_color='#4ECDC4'
        ))
        fig.update_layout(
            barmode='group',
            yaxis_title='F1 Score',
            xaxis_title='Entity Type',
            yaxis_range=[0, 1],
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Precision & Recall table
        st.subheader("Detailed Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Regex**")
            regex_df = pd.DataFrame({
                'Type': entity_types + ['Overall'],
                'P': [regex_results['metrics'][f'{et}_precision'] for et in entity_types] + [regex_results['metrics']['overall_precision']],
                'R': [regex_results['metrics'][f'{et}_recall'] for et in entity_types] + [regex_results['metrics']['overall_recall']],
                'F1': [regex_results['metrics'][f'{et}_f1'] for et in entity_types] + [regex_results['metrics']['overall_f1']]
            })
            st.dataframe(regex_df.style.format({'P': '{:.3f}', 'R': '{:.3f}', 'F1': '{:.3f}'}), use_container_width=True)
        
        with col2:
            st.markdown("**spaCy**")
            spacy_df = pd.DataFrame({
                'Type': entity_types + ['Overall'],
                'P': [spacy_results['metrics'][f'{et}_precision'] for et in entity_types] + [spacy_results['metrics']['overall_precision']],
                'R': [spacy_results['metrics'][f'{et}_recall'] for et in entity_types] + [spacy_results['metrics']['overall_recall']],
                'F1': [spacy_results['metrics'][f'{et}_f1'] for et in entity_types] + [spacy_results['metrics']['overall_f1']]
            })
            st.dataframe(spacy_df.style.format({'P': '{:.3f}', 'R': '{:.3f}', 'F1': '{:.3f}'}), use_container_width=True)
        
        with col3:
            st.markdown(f"**DSPy**")
            dspy_df = pd.DataFrame({
                'Type': entity_types + ['Overall'],
                'P': [dspy_results['metrics'][f'{et}_precision'] for et in entity_types] + [dspy_results['metrics']['overall_precision']],
                'R': [dspy_results['metrics'][f'{et}_recall'] for et in entity_types] + [dspy_results['metrics']['overall_recall']],
                'F1': [dspy_results['metrics'][f'{et}_f1'] for et in entity_types] + [dspy_results['metrics']['overall_f1']]
            })
            st.dataframe(dspy_df.style.format({'P': '{:.3f}', 'R': '{:.3f}', 'F1': '{:.3f}'}), use_container_width=True)
        
        # Cost & Latency KPIs
        st.subheader("Cost & Latency")
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.metric(
                "Cost (Regex)",
                "$0.00",
                help="Regex is free"
            )
        
        with col2:
            st.metric(
                "Cost (spaCy)",
                "$0.00",
                help="spaCy is free"
            )
        
        with col3:
            st.metric(
                "Cost (DSPy)",
                f"${dspy_results['estimated_cost']:.4f}",
                help="Total estimated cost for DSPy"
            )
        
        with col4:
            st.metric(
                "Latency (Regex)",
                format_time(regex_results['avg_latency']),
                help="Average time per sample for regex"
            )
        
        with col5:
            st.metric(
                "Latency (spaCy)",
                format_time(spacy_results['avg_latency']),
                help="Average time per sample for spaCy"
            )
        
        with col6:
            st.metric(
                "Latency (DSPy)",
                format_time(dspy_results['avg_latency']),
                help="Average time per sample for DSPy"
            )
    
    # ===== SAMPLE PREDICTIONS TAB =====
    with tab2:
        st.header("🔍 Sample Predictions")
        
        for i in range(min(10, len(test_samples))):
            with st.expander(f"Sample {i+1}: {test_samples[i]['text'][:80]}..."):
                # Show text
                st.markdown("**📄 Input Text**")
                st.code(test_samples[i]['text'], language=None)
                
                st.markdown("---")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown("**✅ Ground Truth**")
                    for entity_type in ['PER', 'ORG', 'LOC', 'MISC']:
                        entities = test_samples[i]['entities'].get(entity_type, [])
                        display_text = ', '.join(entities) if entities else 'None'
                        st.markdown(f"*{entity_type}:* {display_text}")
                
                with col2:
                    st.markdown("**🔧 Regex**")
                    for entity_type in ['PER', 'ORG', 'LOC', 'MISC']:
                        entities = regex_predictions[i].get(entity_type, [])
                        display_text = ', '.join(entities) if entities else 'None'
                        st.markdown(f"*{entity_type}:* {display_text}")
                
                with col3:
                    st.markdown("**🧠 spaCy**")
                    for entity_type in ['PER', 'ORG', 'LOC', 'MISC']:
                        entities = spacy_predictions[i].get(entity_type, [])
                        display_text = ', '.join(entities) if entities else 'None'
                        st.markdown(f"*{entity_type}:* {display_text}")
                
                with col4:
                    st.markdown(f"**🤖 DSPy**")
                    for entity_type in ['PER', 'ORG', 'LOC', 'MISC']:
                        entities = dspy_predictions[i].get(entity_type, [])
                        display_text = ', '.join(entities) if entities else 'None'
                        st.markdown(f"*{entity_type}:* {display_text}")
    
    # ===== DSPy INTERNALS TAB =====
    with tab3:
        st.header("🔬 DSPy Internals: How the LLM Works")
        
        st.markdown("""
        This tab shows what's happening under the hood when DSPy processes text. You can see:
        - The **prompt** DSPy sends to the LLM
        - The **raw response** from the LLM
        - How DSPy **parses** the response into structured entities
        """)
        
        # Show a few examples
        num_examples = min(5, len(test_samples))
        
        for i in range(num_examples):
            with st.expander(f"Example {i+1}: {test_samples[i]['text'][:80]}...", expanded=(i==0)):
                st.markdown("### 📝 Input Text")
                st.code(test_samples[i]['text'], language=None)
                
                st.markdown("---")
                
                # Show DSPy signature
                st.markdown("### 🎯 DSPy Signature")
                st.code("""
class NERExtractor(dspy.Signature):
    \"\"\"Extract named entities from text and classify them.\"\"\"
    
    text = dspy.InputField(desc="Text to extract entities from")
    people = dspy.OutputField(desc="List of person names (PER)")
    organizations = dspy.OutputField(desc="List of organizations (ORG)")
    locations = dspy.OutputField(desc="List of locations (LOC)")
    miscellaneous = dspy.OutputField(desc="List of other entities (MISC)")
                """, language="python")
                
                st.markdown("---")
                
                # Show ACTUAL LLM prompts from captured history
                if i < len(dspy_histories) and dspy_histories[i].get('messages'):
                    messages = dspy_histories[i]['messages']
                    
                    # Show system message
                    if len(messages) > 0 and messages[0].get('role') == 'system':
                        st.markdown("### 🔧 System Message (DSPy Structure)")
                        st.code(messages[0]['content'], language=None)
                        st.markdown("---")
                    
                    # Show user message (the actual prompt)
                    if len(messages) > 1 and messages[1].get('role') == 'user':
                        st.markdown("### 💬 Prompt sent to LLM")
                        st.code(messages[1]['content'], language=None)
                        st.markdown("---")
                    
                    # Show LLM response
                    if dspy_histories[i].get('outputs'):
                        st.markdown("### 🤖 LLM Response")
                        st.code(dspy_histories[i]['outputs'][0], language=None)
                else:
                    # Fallback to approximate
                    st.markdown("### 💬 Approximate LLM Prompt")
                    st.warning("Could not capture actual prompt from LLM history.")
                    
                    prompt_example = f"""Extract named entities from text and classify them.

---

Follow the following format.

Text: Text to extract entities from
People: List of person names (PER)
Organizations: List of organizations (ORG)
Locations: List of locations (LOC)
Miscellaneous: List of other entities (MISC)

---

Text: {test_samples[i]['text']}
People:"""
                    
                    st.code(prompt_example, language=None)
                
                st.markdown("---")
                
                # Show the prediction
                st.markdown("### 🤖 DSPy Output (Parsed)")
                pred = dspy_predictions[i]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Extracted Entities:**")
                    for entity_type in ['PER', 'ORG', 'LOC', 'MISC']:
                        entities = pred.get(entity_type, [])
                        display_text = ', '.join(entities) if entities else 'None'
                        st.markdown(f"- **{entity_type}**: {display_text}")
                
                with col2:
                    st.markdown("**Ground Truth:**")
                    for entity_type in ['PER', 'ORG', 'LOC', 'MISC']:
                        entities = test_samples[i]['entities'].get(entity_type, [])
                        display_text = ', '.join(entities) if entities else 'None'
                        
                        # Check if correct
                        pred_set = set(pred.get(entity_type, []))
                        true_set = set(entities)
                        is_correct = pred_set == true_set
                        
                        if is_correct:
                            st.markdown(f"- **{entity_type}**: {display_text} ✅")
                        else:
                            st.markdown(f"- **{entity_type}**: {display_text}")
        
        st.markdown("---")
        st.markdown("""
        ### 🔑 Key Insights
        
        - **DSPy Signatures** define the input/output structure without hardcoding prompts
        - **Field Descriptions** guide the LLM on what to extract
        - **Structured Output** is automatically parsed from the LLM's response
        - **No Manual Prompting** - DSPy handles prompt engineering for you
        
        This is why DSPy achieves high accuracy: it uses optimized prompts and structured parsing!
        """)

else:
    st.info("�👈 Configure your experiment in the sidebar and click 'Run Experiment' to get started!")

