# Code accompanying paper titled **Language Models are Crossword Solvers**. 

This folder contains all necessary code to reproduce results in the presented
paper.


## Requirements.

  - Running code in this repository requires a python installation. 
  Tested with ```python==3.12.3```. We recommend a virtual environment
  like ```venv`` or conda.

  - All requisite packages can be installed with the following command.

```
pip install -r requirements.txt

```

## Usage instructions - LLM Inference.
  
  - The first step would be to generate prompts. The code for this can be found
  in the folder ```./Prompts```.
    - Please update the ```./Prompts/config.toml``` file to make sure that the
    paths stored in the config file correspond to the correct path on your system.
    All requisite data is provided in ```Data.tar.gz```.
    - Follow instructions in ```./Prompts/README.md``` to generate prompts.

  - The next step is to generate outputs with LLMs.
    - Please update the ```./LLM-Inference/config.toml``` file to make sure that the
    paths stored in the config file correspond to the correct path on your system for
    the requisite LLMs. If you want Huggingface to download the models for you, please
    just add the name of the model in the corresponding field. We only support local
    inference for models in the Huggingface format (LLaMA models must be converted).
    - Follow instructions in ```./LLM-Inference/README.md``` to generate outputs.
    - Post-processing code is provided in ```./LLM-Inference/src/postprocess.py```.

## Usage instructions - SweepClip.
  - Data is provided in the folder ```./data```. Note this is only provided for
  review and will not be available in the final version, since this data is the
  intellectual property of the New York Times.
  - Follow instructions in ```./SweepClip/README.md``` to generate outputs.
