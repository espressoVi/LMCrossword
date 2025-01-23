# SweepClip

This folder contains code for the SweepClip algorithm.


## Instructions

  - Make sure requirements are installed. Please see ```../README.md```
  - Make sure to edit the configuration file ```config.toml``` to provide the correct path to the dataset
  and local LLaMA 3 70B model.
  - For GPT 4 Turbo please add your API key. This can be done by adding the following line to your environment (bashrc).

```
export OPENAI_API_KEY='your api key'
```
  - Run with the following command.

```
python main.py
```
  - The LLM call logs (queries and outputs) are stored in ```logs``` and the
  generated answers are stored in ```outputs```.

  - For analysis, i.e. calculation of metrics, etc run - 
```
python analysis.py
```
