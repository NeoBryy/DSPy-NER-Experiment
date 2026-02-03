import dspy
import os
from dotenv import load_dotenv

load_dotenv()

MODELS = {
    'gpt-4o-mini': {
        'model': 'gpt-4o-mini',
        'cost_per_1k_input': 0.00015,
        'cost_per_1k_output': 0.0006
    },
    'gpt-4o': {
        'model': 'gpt-4o',
        'cost_per_1k_input': 0.0025,
        'cost_per_1k_output': 0.01
    },
    'o1-mini': {
        'model': 'o1-mini',
        'cost_per_1k_input': 0.003,
        'cost_per_1k_output': 0.012
    }
}

def get_lm(model_name='gpt-4o-mini'):
    config = MODELS[model_name]
    # O-series models only support temperature=1
    temp = 1 if model_name.startswith('o1') else 0.0
    return dspy.LM(
        model=f"openai/{config['model']}",
        api_key=os.getenv('OPENAI_API_KEY'),
        temperature=temp
    )