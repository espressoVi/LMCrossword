from src.llm import LMFactory
import toml
import json
import sys
import os
import getopt
from tqdm import tqdm

config = toml.load("config.toml")


def parse_arguments():
    output_directory, llm_name = "./outputs", "llama3_8Bi"
    argumentList = sys.argv[1:]
    long_options, options = ["help", "input=", "llm="], "hi:l:"
    arguments, values = getopt.getopt(argumentList, options, long_options)
    for currentArgument, currentValue in arguments:
        if currentArgument in ("-h", "--help"):
            print(
                    "\nThis file runs inference with language models given a prompt file and "\
                    "saves the output to a json file. \nUsage: \n"\
                    "\tpython main.py --input [input JSON prompt file] --llm [llm name]\n\n"\
                    "Available LLMs are - {phi3_3Bi, mistral_7Bi, llama3_8Bi, mixtral, "\
                    "llama2_70B, llama3_70B, gpt_3, gpt_4t, claude}.\n"\
            )
            exit(0)
        elif currentArgument in ("-i", "--input"):
            input_file = currentValue
            if not os.path.exists(input_file):
                raise FileNotFoundError(f"File {input_file} does not exist")
            if "json" not in os.path.splitext(input_file)[-1]:
                raise ValueError(f"File {input_file} is not a JSON file.")
        elif currentArgument in ("-l", "--llm"):
            if currentValue not in config['llm']:
                raise ValueError(
                        f"{currentValue} is not an acceptable LLM name"\
                        "Please see 'python main.py --help'."
                )
            llm_name = currentValue
        else:
            raise ValueError("Invalid input, see python main.py --help")
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    if input_file is None:
        raise ValueError("Input file not provided")
    output_file = f"{llm_name}_({os.path.splitext(input_file.split('/')[-1])[0]}).json"
    output_file = os.path.join(output_directory, output_file)
    return llm_name, input_file, output_file

def get_results(llm_name, input_file, output_file):
    with open(input_file, 'r') as f:
        prompts = json.load(f)
    results = {}
    model = LMFactory(llm_name)
    for idx, (tag, prompt) in tqdm(
                                    enumerate(prompts.items()),
                                    desc = "Generating outputs",
                                    total = len(prompts.items(),
                                )
    ):
        answer = model(prompt)[0]
        results[tag] = answer
        #print(f"LLM ANSWER: {answer}")
        if idx%100 == 0 and idx>0:
            write_json(output_file, results)
    write_json(output_file, results)

def write_json(file, dataset):
    with open(file, 'w') as f:
        json.dump(dataset, f, indent = 4)

def main():
    llm_name, input_file, output_file = parse_arguments()
    get_results(llm_name, input_file, output_file)

if __name__ == "__main__":
    main()
