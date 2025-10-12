# Competitive Market Behavior of LLMs

This repository contains code for replicating a behavioural economic experiment
where LLM-powered agents participate in a market.

Note that this code requires an API key for OpenAI.

## Setup

Clone the repository

```bash
git clone https://github.com/your-username/market-equilibrium.git
cd market-equilibrium
```

Create a conda environment from the provided file.

```bash
conda env create -f environment.yml
```

Set an environment variable in your conda environment holding your OpenAI API key
and reload the environment.

```bash
conda env config vars set OPENAI_API_KEY="your_api_key_here"
conda deactivate
conda activate market-equilibrium 
```

## Experiments

To run an experiment, run the `experiment.py` script with one argument indicating 
the name of the config file to use. The config files control all the parameters of
the experiment and are stored in the configs directory. Note that this will call
the OpenAI API which can incur costs. 

```bash
python experiment.py exp2
```

The script will produce a directory named `"{config_name}_{commit_hash}_{timestamp}"`, 
which will contain all the results.

## Results

All the results can be reviewed using the ```results_analysis.ipynb``` notebook.  