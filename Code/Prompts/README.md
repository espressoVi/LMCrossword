# Prompt Generation

This folder contains code to generate few-shot and Chain-of-thought prompts w/o hints.

## Instructions.

  - Make sure requirements are installed. Please see ```../README.md```
  - Make a directory called ```prompts``` at the current location.
  - Edit the ```config.toml``` file to point to the correct file locations.
    All the requisite files are provided in the ```Data``` directory.

## Generating various prompts.
  - Run the following to generate prompts for clue answering - 

```
python Prompt.py

```
  - The ```chain_of_thought``` flag can be set to ```True``` to generate CoT prompts.
  - Outputs are saved in ```./prompts/```.

  - Run the following to generate prompts for hinted clue answering - 

```
python HintedPrompt.py

```
  - Outputs are saved in ```./prompts/```.

  - Run the following to generate prompts for the counting task - 

```
python CountingPrompt.py

```
  - Outputs are saved in ```./prompts/```.
