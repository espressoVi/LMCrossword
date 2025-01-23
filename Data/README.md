# Data accompanying paper titled **Language Models are Crossword Solvers**. 

This folder contains all necessary data to reproduce results in the presented
paper.


## Table of Contents.

Here we provide a list of all data files included and their purpose.

  - **Datasets** - Datasets used to produce results.
    - **Cryptonite-Clues** - Subset of *Cryptonite* dataset used to report results.
    - **Init-Clues** - Subset of *word-init-disjoint* dataset used to report results.
    - **NYT-Clues** - *NYT* dataset clue answer pairs used to report results.
    - **Post_cutoff-Clues** - *Post cutoff* dataset clue answer pairs used to report results.
    - **Sub-token Counting** - The dataset of words used for the counting tests.
      - ```guardian.json``` - Sourced from The Guardian after May 20, 2024.
      - ```lovatts.json``` - Sourced from Lovatts Puzzles after May 20, 2024.
    - **NYT-Grids** - *New York Times* Monday crosswords used for reporting results. Contains a list of dates to
    uniquely identify the crosswords.

  - **Human Eval** - All data pertaining to Human Evaluation.
    - ```Human_Evaluation_Report.pdf``` - All queries, ground truth answers and GPT-4-Turbo outputs used for
    human evaluation. Human annotations are also provided.
    - ```Outputs_For_Human_Eval_GPT_4``` - Raw model outputs.
    - ```post_cutoff_Lovatts_cot_fewShot_3.json``` - Prompt file to generate reasoned responses.

  - **Prompts** - All prompts used for generating LLM completions. The prompts are placed in folders
  corresponding to the paper sections in which they are used.

  - **Raw Outputs** - Raw model outputs from LLMs for each section.

  - ```CoT_Solved_Examples.json``` - Manually curated set of solved examples for *Chain of Thought*.

  - ```README.md``` - This file.
