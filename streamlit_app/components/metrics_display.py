"""
Metrics display component for NER experiment results.
Renders F1 scores, charts, detailed metrics tables, and cost/latency information.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd


def format_time(seconds):
    """Format time duration in human-readable format."""
    if seconds < 0.001:
        return f"{seconds * 1000000:.0f}µs"
    elif seconds < 1:
        return f"{seconds * 1000:.2f}ms"
    else:
        return f"{seconds:.2f}s"


def render(regex_results, spacy_results, dspy_results, extractor_name, use_implicit):
    """
    Render metrics comparison tab.
    
    Args:
        regex_results: dict - Regex baseline results
        spacy_results: dict - spaCy baseline results
        dspy_results: dict - DSPy model results
        extractor_name: str - Name of DSPy extractor variant
        use_implicit: bool - Whether implicit mode is enabled
    """
    st.header("📊 Performance Comparison")
    
    # Show appropriate F1 metrics based on mode
    if use_implicit:
        _render_implicit_metrics(regex_results, spacy_results, dspy_results, extractor_name)
    else:
        _render_standard_metrics(regex_results, spacy_results, dspy_results)
    
    st.markdown("---")
    
    # F1 by entity type charts
    if use_implicit:
        _render_implicit_entity_chart(regex_results, spacy_results, dspy_results, extractor_name)
    else:
        _render_standard_entity_chart(regex_results, spacy_results, dspy_results)
    
    # Detailed metrics tables (only for standard mode)
    if not use_implicit:
        _render_detailed_metrics(regex_results, spacy_results, dspy_results, use_implicit)
    
    # Cost & Latency KPIs
    _render_cost_latency(regex_results, spacy_results, dspy_results)


def _render_implicit_metrics(regex_results, spacy_results, dspy_results, extractor_name):
    """Render explicit and implicit F1 metrics for implicit mode."""
    st.subheader("Explicit Entity Extraction (Sentence 1)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "🔧 Regex",
            f"{regex_results['metrics']['overall_explicit_f1']:.1%}",
            help="Rule-based pattern matching"
        )
    
    with col2:
        delta_vs_regex = spacy_results['metrics']['overall_explicit_f1'] - regex_results['metrics']['overall_explicit_f1']
        st.metric(
            "🧠 spaCy",
            f"{spacy_results['metrics']['overall_explicit_f1']:.1%}",
            delta=f"{delta_vs_regex:+.1%} from Regex",
            help="Traditional ML model"
        )
    
    with col3:
        delta_vs_regex = dspy_results['metrics']['overall_explicit_f1'] - regex_results['metrics']['overall_explicit_f1']
        st.metric(
            f"🤖 {extractor_name}",
            f"{dspy_results['metrics']['overall_explicit_f1']:.1%}",
            delta=f"{delta_vs_regex:+.1%} from Regex",
            help="LLM-powered extraction"
        )
    
    st.markdown("---")
    
    st.subheader("Implicit Resolution (Sentence 2)")
    st.caption("Can the model extract references like 'He', 'The city', 'The company'?")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "🔧 Regex",
            f"{regex_results['metrics']['overall_implicit_f1']:.1%}",
            help="Cannot resolve implicit references"
        )
    
    with col2:
        st.metric(
            "🧠 spaCy",
            f"{spacy_results['metrics']['overall_implicit_f1']:.1%}",
            help="Cannot resolve implicit references"
        )
    
    with col3:
        improvement = dspy_results['metrics']['overall_implicit_f1']
        st.metric(
            f"🤖 {extractor_name}",
            f"{dspy_results['metrics']['overall_implicit_f1']:.1%}",
            delta=f"+{improvement:.1%} vs baselines",
            help="LLM with proper prompting can extract implicit references!"
        )


def _render_standard_metrics(regex_results, spacy_results, dspy_results):
    """Render overall F1 metrics for standard mode."""
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


def _render_implicit_entity_chart(regex_results, spacy_results, dspy_results, extractor_name):
    """Render implicit F1 by entity type chart."""
    st.subheader("Implicit F1 by Entity Type (DSPy Model)🔍")
    
    entity_types = ['PER', 'ORG', 'LOC', 'MISC']
    regex_implicit_f1s = [regex_results['metrics'].get(f'implicit_{et}_f1', 0) for et in entity_types]
    spacy_implicit_f1s = [spacy_results['metrics'].get(f'implicit_{et}_f1', 0) for et in entity_types]
    dspy_implicit_f1s = [dspy_results['metrics'].get(f'implicit_{et}_f1', 0) for et in entity_types]
    
    fig = go.Figure()
    # Only show DSPy for implicit mode since baselines are always 0
    fig.add_trace(go.Bar(
        name=extractor_name,
        x=entity_types,
        y=dspy_implicit_f1s,
        marker_color='#4ECDC4'
    ))
    fig.update_layout(
        title=f"{extractor_name} Performance by Entity Type",
        yaxis_title='Implicit F1 Score',
        xaxis_title='Entity Type',
        yaxis_range=[0, 1],
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Add insight box
    st.info(f"""
    💡 **Key Insight**: {extractor_name} achieves {dspy_results['metrics']['overall_implicit_f1']:.1%} implicit F1!
    Standard models (Regex, spaCy) score 0% because they cannot resolve references without explicit names.
    """)


def _render_standard_entity_chart(regex_results, spacy_results, dspy_results):
    """Render F1 by entity type chart for standard mode."""
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


def _render_detailed_metrics(regex_results, spacy_results, dspy_results, use_implicit=False):
    """Render detailed precision/recall/F1 tables."""
    st.subheader("Detailed Metrics")
    
    # In implicit mode, show explicit metrics (sentence 1 performance)
    # In standard mode, show overall metrics
    if use_implicit:
        st.caption("Explicit entity extraction performance (Sentence 1 only)")
    
    entity_types = ['PER', 'ORG', 'LOC', 'MISC']
    col1, col2, col3 = st.columns(3)
    
    # Determine metric key prefix based on mode
    prefix = 'explicit_' if use_implicit else ''
    
    with col1:
        st.markdown("**Regex**")
        regex_df = pd.DataFrame({
            'Type': entity_types + ['Overall'],
            'P': [regex_results['metrics'].get(f'{prefix}{et}_precision', 0) for et in entity_types] + [regex_results['metrics'].get(f'{prefix}overall_precision', regex_results['metrics'].get('overall_precision', 0))],
            'R': [regex_results['metrics'].get(f'{prefix}{et}_recall', 0) for et in entity_types] + [regex_results['metrics'].get(f'{prefix}overall_recall', regex_results['metrics'].get('overall_recall', 0))],
            'F1': [regex_results['metrics'].get(f'{prefix}{et}_f1', 0) for et in entity_types] + [regex_results['metrics'].get(f'{prefix}overall_f1', regex_results['metrics'].get('overall_f1', 0))]
        })
        st.dataframe(regex_df.style.format({'P': '{:.3f}', 'R': '{:.3f}', 'F1': '{:.3f}'}), use_container_width=True)
    
    with col2:
        st.markdown("**spaCy**")
        spacy_df = pd.DataFrame({
            'Type': entity_types + ['Overall'],
            'P': [spacy_results['metrics'].get(f'{prefix}{et}_precision', 0) for et in entity_types] + [spacy_results['metrics'].get(f'{prefix}overall_precision', spacy_results['metrics'].get('overall_precision', 0))],
            'R': [spacy_results['metrics'].get(f'{prefix}{et}_recall', 0) for et in entity_types] + [spacy_results['metrics'].get(f'{prefix}overall_recall', spacy_results['metrics'].get('overall_recall', 0))],
            'F1': [spacy_results['metrics'].get(f'{prefix}{et}_f1', 0) for et in entity_types] + [spacy_results['metrics'].get(f'{prefix}overall_f1', spacy_results['metrics'].get('overall_f1', 0))]
        })
        st.dataframe(spacy_df.style.format({'P': '{:.3f}', 'R': '{:.3f}', 'F1': '{:.3f}'}), use_container_width=True)
    
    with col3:
        st.markdown(f"**DSPy**")
        dspy_df = pd.DataFrame({
            'Type': entity_types + ['Overall'],
            'P': [dspy_results['metrics'].get(f'{prefix}{et}_precision', 0) for et in entity_types] + [dspy_results['metrics'].get(f'{prefix}overall_precision', dspy_results['metrics'].get('overall_precision', 0))],
            'R': [dspy_results['metrics'].get(f'{prefix}{et}_recall', 0) for et in entity_types] + [dspy_results['metrics'].get(f'{prefix}overall_recall', dspy_results['metrics'].get('overall_recall', 0))],
            'F1': [dspy_results['metrics'].get(f'{prefix}{et}_f1', 0) for et in entity_types] + [dspy_results['metrics'].get(f'{prefix}overall_f1', dspy_results['metrics'].get('overall_f1', 0))]
        })
        st.dataframe(dspy_df.style.format({'P': '{:.3f}', 'R': '{:.3f}', 'F1': '{:.3f}'}), use_container_width=True)


def _render_cost_latency(regex_results, spacy_results, dspy_results):
    """Render cost and latency metrics."""
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
        # Calculate cache hit rate if available
        cache_info = ""
        if 'token_stats' in dspy_results and dspy_results['token_stats']['total_prompt_tokens'] > 0:
            cache_rate = (dspy_results['token_stats']['total_cached_tokens'] / 
                         dspy_results['token_stats']['total_prompt_tokens'])
            if cache_rate > 0:
                cache_info = f" ({cache_rate:.0%} cached)"
        
        st.metric(
            "Cost (DSPy)",
            f"${dspy_results['estimated_cost']:.4f}{cache_info}",
            help="Total estimated cost for DSPy. OpenAI's prompt caching reduces costs by ~50% on cached tokens (requires 1024+ token prefix)."
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
