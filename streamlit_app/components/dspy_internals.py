"""
DSPy internals component for showing LLM prompts and responses.
Displays how DSPy constructs prompts and parses responses.
"""

import streamlit as st


def render(dspy_histories):
    """
    Render DSPy internals tab showing prompts and responses.
    
    Args:
        dspy_histories: list - Captured LLM interaction histories
    """
    st.header("🔬 DSPy Internals: How the LLM Works")
    
    st.markdown("""
    This tab shows what's happening under the hood when DSPy processes text. You can see:
    - The **prompt** DSPy sends to the LLM
    - The **raw response** from the LLM
    - How DSPy **parses** the response into structured entities
    """)
    
    # Show a few examples
    num_examples = min(5, len(dspy_histories))
    
    for i in range(num_examples):
        history = dspy_histories[i]
        with st.expander(f"Example {i+1}: {history['text'][:80]}...", expanded=(i==0)):
            st.markdown("### 📝 Input Text")
            st.code(history['text'], language=None)
            
            st.markdown("---")
            
            # Show DSPy signature
            st.markdown("### 🎯 DSPy Signature")
            st.code("""
class NERExtractor(dspy.Signature):
    \\\"\\\"\\\"Extract named entities from text and classify them.\\\"\\\"\\\"
    
    text = dspy.InputField(desc="Text to extract entities from")
    people = dspy.OutputField(desc="List of person names (PER)")
    organizations = dspy.OutputField(desc="List of organizations (ORG)")
    locations = dspy.OutputField(desc="List of locations (LOC)")
    miscellaneous = dspy.OutputField(desc="List of other entities (MISC)")
            """, language="python")
            
            st.markdown("---")
            
            # Show ACTUAL LLM prompts from captured history
            if history.get('messages'):
                messages = history['messages']
                
                # Combine system message + user input into unified prompt display
                st.markdown("### 💬 Prompt sent to LLM")
                
                # Start with system message if available
                if len(messages) > 0 and messages[0].get('role') == 'system':
                    system_content = messages[0]['content']
                    # Replace the {text} placeholder with actual input
                    # Extract the template part before the examples/input
                    prompt_parts = [system_content]
                    
                    # Add the actual input text
                    prompt_parts.append(f"\n[[ ## text ## ]]\n{history['text']}")
                    
                    # Add the output template (empty placeholders)
                    if 'reasoning' in system_content:
                        prompt_parts.append("\n[[ ## reasoning ## ]]\n{reasoning}")
                    prompt_parts.append("\n[[ ## people ## ]]\n{people}")
                    prompt_parts.append("\n[[ ## organizations ## ]]\n{organizations}")
                    prompt_parts.append("\n[[ ## locations ## ]]\n{locations}")
                    prompt_parts.append("\n[[ ## misc ## ]]\n{misc}")
                    prompt_parts.append("\n[[ ## completed ## ]]")
                    
                    if 'reasoning' in system_content:
                        prompt_parts.append("\nIn adhering to this structure, your objective is: \n        Extract named entities with reasoning about implicit references.")
                    else:
                        prompt_parts.append("\nIn adhering to this structure, your objective is: \n        Extract named entities from text and classify them.")
                    
                    st.code(''.join(prompt_parts), language=None)
                else:
                    # Fallback: just show the input
                    st.code(f"[[ ## text ## ]]\n{history['text']}", language=None)
                
                # Show LLM response
                if history.get('outputs'):
                    st.markdown("### 🤖 LLM Response")
                    st.code(history['outputs'][0], language=None)
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

Text: {history['text']}
People:"""
                
                st.code(prompt_example, language=None)
            
            st.markdown("---")
            
            # Show the prediction
            st.markdown("### 🤖 DSPy Output (Parsed)")
            pred = history['prediction']
            
            st.markdown("**Extracted Entities:**")
            for entity_type in ['PER', 'ORG', 'LOC', 'MISC']:
                entities = pred.get(entity_type, [])
                display_text = ', '.join(entities) if entities else 'None'
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
