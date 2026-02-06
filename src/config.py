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
    'qwen-80b': {
        'model': 'lm_studio/qwen/qwen3-next-80b',
        'cost_per_1k_input': 0,
        'cost_per_1k_output': 0
    }
}

def get_lm(model_name='qwen-80b'):
    config = MODELS[model_name]
    return dspy.LM(
        model=config['model'],
        #provider='lm_studio',
        api_base="http://framework.tawny-bellatrix.ts.net:1234/v1/",
        #api_key="home",
        #temperature=0.0
    )