"""
Sidebar component for NER experiment configuration.
Handles model selection, NER mode, CoT/Few-Shot options, and cost estimation.
"""

import streamlit as st
from src.config import MODELS


def render_sidebar():
    """
    Render the sidebar with all configuration options.
    
    Returns:
        dict: Configuration dictionary with keys:
            - model_name: str
            - use_implicit: bool
            - use_cot: bool
            - use_fewshot: bool
            - sample_size: int
            - model_config: dict
            - estimated_cost: float
    """
    st.sidebar.header("⚙️ Experiment Configuration")
    
    # Model selection
    model_name = st.sidebar.selectbox(
        "Select LLM Model",
        options=list(MODELS.keys()),
        index=0,
        help="Choose which OpenAI model to use for DSPy extraction"
    )
    
    # NER Mode selection
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 NER Mode")
    ner_mode = st.sidebar.radio(
        "Select extraction mode",
        options=["Standard (Explicit Entities)", "Implicit Resolution"],
        help="Standard: Extract explicitly mentioned entities\\nImplicit: Extract pronouns and references like 'He', 'The city'"
    )
    
    # Show implicit options if implicit mode selected
    use_implicit = "Implicit" in ner_mode
    if use_implicit:
        st.sidebar.markdown("**Implicit Enhancement Options:**")
        
        use_cot = st.sidebar.checkbox(
            "Enable Chain-of-Thought",
            value=True,
            help="Asks the LLM to reason step-by-step before answering. Example: '1) Find explicit entities, 2) Identify pronouns like He/She, 3) List both'"
        )
        
        use_fewshot = st.sidebar.checkbox(
            "Enable Few-Shot Examples",
            value=True,
            help="Provides demonstration examples showing implicit extraction. Example: 'Tim Cook announced products. He praised the team.' → Extract: ['Tim Cook', 'He']"
        )
    else:
        use_cot = False
        use_fewshot = False
    
    st.sidebar.markdown("---")
    
    # Sample size
    sample_size = st.sidebar.slider(
        "Records to Test",
        min_value=5,
        max_value=100 if not use_implicit else 50,
        value=20,
        step=5,
        help="More records = more accurate results but higher cost"
    )
    
    # Display model info with estimated cost
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Model Info")
    model_config = MODELS[model_name]
    
    # Estimate cost based on typical token usage
    estimated_cost = calculate_estimated_cost(
        use_implicit, use_cot, use_fewshot, sample_size, model_config
    )
    
    st.sidebar.write(f"**Estimated Cost:** ${estimated_cost:.4f}")
    mode_text = "implicit" if use_implicit else "standard"
    st.sidebar.caption(f"Based on {sample_size} {mode_text} samples with {model_name}")
    
    return {
        'model_name': model_name,
        'use_implicit': use_implicit,
        'use_cot': use_cot,
        'use_fewshot': use_fewshot,
        'sample_size': sample_size,
        'model_config': model_config,
        'estimated_cost': estimated_cost
    }


def calculate_estimated_cost(use_implicit, use_cot, use_fewshot, sample_size, model_config):
    """
    Calculate estimated API cost based on configuration.
    
    Args:
        use_implicit: bool - Whether implicit mode is enabled
        use_cot: bool - Whether Chain-of-Thought is enabled
        use_fewshot: bool - Whether Few-Shot is enabled
        sample_size: int - Number of samples
        model_config: dict - Model pricing configuration
    
    Returns:
        float: Estimated cost in USD
    """
    # Standard mode: ~100 input, ~50 output tokens per sample
    # Implicit mode: higher token usage due to multi-sentence context
    # CoT adds reasoning tokens, Few-Shot adds example tokens
    if use_implicit:
        if use_cot and use_fewshot:
            avg_input_tokens = 250 * sample_size  # CoT + Few-Shot examples + multi-sentence
            avg_output_tokens = 80 * sample_size   # Reasoning + entities
        elif use_cot:
            avg_input_tokens = 200 * sample_size  # CoT instructions + multi-sentence
            avg_output_tokens = 70 * sample_size   # Reasoning + entities
        elif use_fewshot:
            avg_input_tokens = 220 * sample_size  # Few-Shot examples + multi-sentence
            avg_output_tokens = 60 * sample_size   # Entities
        else:
            avg_input_tokens = 150 * sample_size  # Just multi-sentence
            avg_output_tokens = 50 * sample_size
    else:
        avg_input_tokens = 100 * sample_size
        avg_output_tokens = 50 * sample_size
    
    estimated_cost = (
        (avg_input_tokens / 1000) * model_config['cost_per_1k_input'] +
        (avg_output_tokens / 1000) * model_config['cost_per_1k_output']
    )
    
    return estimated_cost
