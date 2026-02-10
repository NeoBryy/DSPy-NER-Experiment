"""
Sample viewer component for displaying model predictions.
Shows side-by-side comparison of ground truth vs model outputs with implicit ref highlighting.
"""

import streamlit as st


def render(test_samples, regex_predictions, spacy_predictions, dspy_predictions, extractor_name, use_implicit):
    """
    Render sample predictions tab.
    
    Args:
        test_samples: list - Test samples with ground truth
        regex_predictions: list - Regex predictions
        spacy_predictions: list - spaCy predictions
        dspy_predictions: list - DSPy predictions
        extractor_name: str - Name of DSPy extractor variant
        use_implicit: bool - Whether implicit mode is enabled
    """
    st.header("🔍 Sample Predictions")
    
    if use_implicit:
        st.caption("Multi-sentence samples: Sentence 1 has explicit entities, Sentence 2 has implicit references")
    
    for i in range(min(10, len(test_samples))):
        with st.expander(f"Sample {i+1}: {test_samples[i]['text'][:80]}..."):
            # Show text
            st.markdown("**📄 Input Text**")
            st.code(test_samples[i]['text'], language=None)
            
            # Highlight implicit references if in implicit mode
            if use_implicit and 'implicit_refs' in test_samples[i]:
                st.markdown("**🎯 Implicit References in this sample:**")
                refs_text = ", ".join([f"'{ref['text']}' ({ref['type']})" for ref in test_samples[i]['implicit_refs']])
                st.info(f"Implicit refs to extract: {refs_text}")
            
            st.markdown("---")
            
            col1, col2, col3, col4 = st.columns(4)
            
            # Ground Truth
            with col1:
                st.markdown("**✅ Ground Truth**")
                if use_implicit:
                    _render_implicit_entities(test_samples[i], is_ground_truth=True)
                else:
                    _render_standard_entities(test_samples[i]['entities'])
            
            # Regex
            with col2:
                st.markdown("**🔧 Regex**")
                _render_standard_entities(regex_predictions[i])
            
            # spaCy
            with col3:
                st.markdown("**🧠 spaCy**")
                _render_standard_entities(spacy_predictions[i])
            
            # DSPy
            with col4:
                st.markdown(f"**🤖 {extractor_name}**")
                if use_implicit:
                    _render_implicit_entities_dspy(test_samples[i], dspy_predictions[i])
                else:
                    _render_standard_entities(dspy_predictions[i])


def _render_standard_entities(entities):
    """Render entities for standard mode."""
    for entity_type in ['PER', 'ORG', 'LOC', 'MISC']:
        entity_list = entities.get(entity_type, [])
        display_text = ', '.join(entity_list) if entity_list else 'None'
        st.markdown(f"*{entity_type}:* {display_text}")


def _render_implicit_entities(sample, is_ground_truth=False):
    """Render entities with implicit references highlighted."""
    for entity_type in ['PER', 'ORG', 'LOC', 'MISC']:
        # 1. Get explicit entities from Sentence 1
        explicit_entities = sample.get('sentence1_entities', {}).get(entity_type, [])
        
        # 2. Get implicit reference TEXTS (e.g., "The city", "He")
        implicit_refs = [ref['text'] for ref in sample.get('implicit_refs', []) 
                        if ref['type'] == entity_type]
        
        # Combine them
        all_entities = explicit_entities + [f"**{ref}** 🔗" for ref in implicit_refs]
        
        if all_entities:
            display_text = ', '.join(all_entities)
        else:
            display_text = 'None'
        
        st.markdown(f"*{entity_type}:* {display_text}")


def _render_implicit_entities_dspy(sample, prediction):
    """Render DSPy predictions with implicit refs correctly extracted marked."""
    for entity_type in ['PER', 'ORG', 'LOC', 'MISC']:
        entities = prediction.get(entity_type, [])
        
        # Highlight implicit refs that were correctly extracted
        implicit_texts = {ref['text'] for ref in sample.get('implicit_refs', []) if ref['type'] == entity_type}
        
        if entities:
            marked_entities = []
            for ent in entities:
                if ent in implicit_texts:
                    marked_entities.append(f"**{ent}** ✅")  # Correctly extracted implicit!
                else:
                    marked_entities.append(ent)
            display_text = ', '.join(marked_entities)
        else:
            display_text = 'None'
        
        st.markdown(f"*{entity_type}:* {display_text}")
