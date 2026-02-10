import dspy
import os
from dotenv import load_dotenv

load_dotenv()

MODELS = {
    'gpt-4o-mini': {
        'provider': 'openai',
        'model': 'gpt-4o-mini',
        'cost_per_1k_input': 0.00015,
        'cost_per_1k_output': 0.0006
    },
    'gpt-4o': {
        'provider': 'openai',
        'model': 'gpt-4o',
        'cost_per_1k_input': 0.0025,
        'cost_per_1k_output': 0.01
    },
    'qwen-80b': {
        'provider': 'lm_studio',
        'model': 'qwen/qwen3-next-80b',
        'cost_per_1k_input': 0,
        'cost_per_1k_output': 0
    },
    'qwen3-8b': {
        'provider': 'lm_studio',
        'model': 'qwen/qwen3-8b',
        'cost_per_1k_input': 0,
        'cost_per_1k_output': 0
    }
}

def get_lm(model_name='qwen3-8b'):
    config = MODELS[model_name]
    if config['provider'] == 'lm_studio':
        api_key='local'
    else:
        api_key=os.getenv('OPENAI_API_KEY'),

    lm = dspy.LM(
        api_base="http://framework.tawny-bellatrix.ts.net:1234/v1/",
        model=f"{config['provider']}/{config['model']}",
        api_key=api_key,
        temperature=0.0
    )
    # Disable DSPy's internal cache to get actual OpenAI usage data (including cached_tokens)
    lm.cache = False
    return lm