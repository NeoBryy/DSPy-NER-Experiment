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
    }
}

def get_lm(model_name='gpt-4o-mini'):
    config = MODELS[model_name]
    return dspy.LM(
        model=f"openai/{config['model']}",
        api_key=os.getenv('OPENAI_API_KEY'),
        temperature=0.0
    )