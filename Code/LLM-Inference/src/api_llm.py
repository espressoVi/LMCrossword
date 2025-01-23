import toml
from openai import OpenAI
from anthropic import Anthropic
from time import sleep

config = toml.load("config.toml")


class GPT:
    def __init__(self, name = "gpt_3"):
        self.name = name
        self.model = config['llm'][name]
        self.client = OpenAI()

    def __call__(self, prompt:list[dict]) -> list[str]:
        """ 
        Calls the OpenAI API and returns completions. The prompt must be in
        the following format:

        prompt = [
              {"role": "system", "content": config['prompt']['system']},
              {"role": "user", "content": prompt},
        ]
        Args:
            list[dict] : See docstring of infer method for explanation.
        Returns:
            list[str]: Completions by GPT.
        """
        completion = self.client.chat.completions.create(
                  model = self.model,
                  messages= prompt,
                  max_tokens = config['gen_config']['max_new'],
                  n = config['gen_config']['seq_num'],
                  temperature = config['gen_config']['temperature'],
                  logprobs = True,
                  top_logprobs = 5,
                  top_p = None,
        )
        return [i.message.content for i in completion.choices]

class Claude:
    def __init__(self, name = "claude"):
        self.name = name
        self.model = config['llm'][name]
        self.client = Anthropic()

    def __call__(self, prompt:list[dict]) -> list[str]:
        """ 
        Calls the OpenAI API and returns completions. The prompt must be in
        the following format:

        prompt = [
              {"role": "system", "content": config['prompt']['system']},
              {"role": "user", "content": prompt},
        ]
        Args:
            list[dict] : See docstring of infer method for explanation.
        Returns:
            list[str]: Completions by GPT.
        """
        results = []
        for i in range(config['gen_config']['seq_num']):
            completion = self.client.messages.create(
                    model = self.model,
                    max_tokens = config['gen_config']['max_new'],
                    temperature = config['gen_config']['temperature'],
                    system = prompt[0]['content'],
                    messages = [prompt[-1]]
            )
            results.append(completion.content[0].text)
        return results if results else [""]
