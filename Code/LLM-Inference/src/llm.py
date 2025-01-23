import toml
from src.local_llm import *
from src.api_llm import GPT, Claude


def LMFactory(name: str) -> transformers.AutoModel:
    if name not in config['llm']:
        raise ValueError(
                f"{name} is not a valid LLM name. "
                f"Please choose from {list(config['llm'].keys())}."
        )
    if 'gpt' in name:
        return GPT(name)
    elif 'claude' in name:
        return Claude()
    else:
        if not os.path.exists(config['llm'][name]):
            return Dummy(name)
        return LLM(name)
