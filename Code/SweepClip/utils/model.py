from openai import OpenAI
import warnings
import torch
import json
import os
import accelerate
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import toml


config = toml.load("config.toml")


class LLM:
    def __init__(self, model):
        assert model in ['gpt4', 'llama3']
        self.cost = 0
        self.cost_threshold = config[model]['budget']

    @property
    def budget_exceeded(self):
        return (self.cost > self.cost_threshold)

    def reset_cost(self):
        self.cost = 0

    def process(self, pred):
        ans = pred.strip().lower()
        ans = ans.replace("clue: ","")
        ans = ans.replace("answer: ","")
        ans = [i for i in ans.split("\n") if i]
        ans = "" if not ans else ans[0]
        ans = "".join([i for i in ans if i.isalpha()])
        return ans

    def infer(self, prompt):
        raise NotImplementedError

    def __call__(self, prompt:list[dict]) -> list[str]:
        """ 
        Calls the LLM and returns completions. The prompt must be in
        the following format:

        prompt = [
              {"role": "system", "content": config['prompt']['system']},
              {"role": "user", "content": prompt},
        ]
        Args:
            list[dict] : See docstring of infer method for explanation.
        Returns:
            list[str]: Completions by GPT.
            bool: Budget exceeded.
        """
        if self.budget_exceeded:
            return [""], True
        response = self.infer(prompt)
        log = f"{prompt[-1]['content'].split("\n")[-1]} {response[0]}\n"
        with open(f"./logs/Crossword_all.txt", "a") as f:
            f.write(log)
        return response, self.budget_exceeded


class GPT(LLM):
    def __init__(self):
        super().__init__('gpt4')
        self.model = config['gpt4']['name']
        self.client = OpenAI()

    def infer(self, prompt):
        cmpl = self.client.chat.completions.create(
                  model = self.model,
                  messages= prompt,
                  max_tokens = config['model']['max_new'],
                  n = config['model']['seq_num'],
                  temperature = config['model']['temperature'],
                  top_p = None,
        )
        self.cost += 1e-6*(cmpl.usage.completion_tokens*config['gpt4']['output_cost']
                      +cmpl.usage.prompt_tokens*config['gpt4']['input_cost'])
        return [self.process(i.message.content) for i in cmpl.choices]


class LLaMA(LLM):
    """ Base LLM class implements inference """
    def __init__(self):
        super().__init__('llama3')
        self.llm_path = config['llama3']['path']
        self.loaded = False
        self.load_tokenizer()
        self._init_config()

    def load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_path, trust_remote_code = True)
        self.tokenizer.unk_token = "<notrequired>" 
        self.tokenizer.sep_token = "<notrequired>"
        self.tokenizer.pad_token = "<notrequired>"
        self.tokenizer.cls_token = "<notrequired>"
        self.tokenizer.mask_token = "<notrequired>"
        self.tokenizer.chat_template = config['llama3']['template']

    def load_model(self):
        if self.loaded:
            return
        self.model = AutoModelForCausalLM.from_pretrained(
                    self.llm_path,
                    torch_dtype = torch.bfloat16,
                    device_map = "auto",
                    trust_remote_code = True,
                )
        self.loaded = True

    def _init_config(self):
        """ Inititalize generation config. """
        self.generation_config = transformers.GenerationConfig(
                max_new_tokens = config['model']['max_new'],
                temperature=config['model']['temperature'],
                top_p = None,
                do_sample = True,
                num_return_sequences = config['model']['seq_num'],
                pad_token_id = self.tokenizer.eos_token_id,
                eos_token_id = self.tokenizer.eos_token_id,
        )


    def infer(self, prompt):
        self.load_model()
        torch.cuda.empty_cache()
        self.model.eval()
        inputs = self.tokenizer.apply_chat_template(prompt, tokenize=True, return_tensors="pt")
        prompt_length = len(self.tokenizer.decode(inputs[0], skip_special_tokens=False))
        with torch.inference_mode():
            results = self.model.generate(
                    input_ids = inputs.to("cuda"),
                    generation_config = self.generation_config,
            )
        results = [self.tokenizer.decode(result, skip_special_tokens=False) for result in results]
        results = [result[prompt_length:] for result in results]
        self.cost += 1
        return [self.process(result) for result in results]
