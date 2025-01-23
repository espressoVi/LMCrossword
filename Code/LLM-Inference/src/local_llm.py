import torch
import toml
import json
import os
import accelerate
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

config = toml.load("config.toml")


class Dummy:
    """
    Dummy LLM class to test prompt pipeline and handle
    path not found exceptions.
    """
    def __init__(self, *args):
        self.args = args

    def __call__(self, prompt):
        return [f"The LLM - {self.args[0]} is not present on this machine"]

class LLM:
    """ Base LLM class implements inference """
    def __init__(self, name):
        self.name = name
        self.llm_path = config['llm'][name]
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
        if self.name in config['templates']:
            self.tokenizer.chat_template = config['templates'][self.name]

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
                max_new_tokens = config['gen_config']['max_new'],
                temperature=config['gen_config']['temperature'],
                top_p = None,
                do_sample = True,
                num_return_sequences = config['gen_config']['seq_num'],
                pad_token_id = self.tokenizer.eos_token_id,
                eos_token_id = self.tokenizer.eos_token_id,
        )

    def __call__(self, prompt:list[dict]) -> list[str]:
        """ 
        Run inference and returns completions. The prompt must be in
        the following format:

        prompt = [
              {"role": "system", "content": config['prompt']['system']},
              {"role": "user", "content": prompt},
        ]
        -------------------
        Args:
            list[dict] : See docstring of infer method for explanation.
        Returns:
            list[str]: Completions by GPT.
        """
        self.load_model()
        torch.cuda.empty_cache()
        self.model.eval()
        inputs = self.tokenizer.apply_chat_template(prompt, tokenize=True, return_tensors="pt")
        prompt_length = len(self.tokenizer.decode(inputs[0], skip_special_tokens=False))
        #print(self.tokenizer.decode(inputs[0]))
        with torch.inference_mode():
            results = self.model.generate(
                    input_ids = inputs.to("cuda"),
                    generation_config = self.generation_config,
            )
        results = [self.tokenizer.decode(result, skip_special_tokens=False) for result in results]
        results = [result[prompt_length:] for result in results]
        return results
