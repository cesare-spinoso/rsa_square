# (RSA)²: A Rhetorical-Strategy-Aware Rational Speech Act Framework for Figurative Language Understanding

Welcome to this code repository! You will find the code and datasets used in the paper [(RSA)²: A Rhetorical-Strategy-Aware Rational Speech Act Framework for Figurative Language Understanding](https://aclanthology.org/2025.acl-long.1019/) (Spinoso-Di Piano et al., ACL 2025) presented in Vienna at ACL 2025.

# Datasets

**Experimental data and results:** We have uploaded all of our experimental datasets, results and analyses to this repository under the `data/` folder. We have provided instructions below and in the rest of the repo on how to use the datasets in the `data/` folder to replicate our experiments and results.

**PragMega+ Hugging Face Dataset:** We have also created a Hugging Face 🤗 Hub dataset for our new **PragMega+** irony understanding dataset at [cesare-spinoso/PragMegaPlus](https://huggingface.co/datasets/cesare-spinoso/PragMegaPlus). To use it, you can use the following `python` snippet:

```
from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("cesare-spinoso/PragMegaPlus")
```

# Code

**File structure:** This repository is organized in the following way:

```
rsa_square/
├── pyproject.toml # Repo + environment installation
├── README.md # **You are here!**
├── data/ # Experimental data, results and analyses (stored with Git-LFS)
│   ├── exp_data/ # Experimental data used for running experiments
│   ├── nonliteral_number_exps/ # Results for the non-literal number expressions
│   ├── ironic_weather_utterances_exps/ # Results for the ironic weather utterance experiments
│   └── prag_mega_plus_exps/ # Results for PragMega+ experiments
└── src/
    ├── configs/ # Hydra configs
    ├── llms/ # LLM utils
    ├── utils/ # Misc utils
    ├── run.py # Script for generating LLM-related outputs (e.g., alternative utterances, prior probability computation)
    ├── rsa.py # Script for applying RSA equations on LLM-generated outputs
    ├── run_all_rsa.py # Script for running rsa.py on multiple configs
    ├── results_and_analysis.py # Script for generating all the results (tables, plots) and analyses (ablations) of our PragMega+-related experiments
    # Notebook for reproducing appendix results
    ├── appendix_table_5_and_6.ipynb
    ├── appendix_table_7.ipynb
    # Non-literal number and ironic weather utterance experiments + results scripts
    ├── nonliteral_numbers_exp.py
    └── weather_utterances_exp.py
```

**Repository installation:** If you want to run our code, follow the steps below which install the repository + python environment.

0. **Install the repository + python environment.** We use [Git-LFS](https://git-lfs.com/) to store our experimental data and results. As a result, when you issue the `git clone` command, it **will not automatically download the experimental data and results**. To do so, you need to use `git clone` and `git lfs` together:

```
git clone https://github.com/cesare-spinoso/rsa_square.git
git lfs pull -I "data/**"
```

where the second line pulls the files from the LFS storage into the `data/` folder. This assumes that you have `git lfs` installed. If you don't, [this guide will help you with the installation.](https://github.com/git-lfs/git-lfs?tab=readme-ov-file#installing).

1. Create and activate a `conda` environment using `python 3.11`

```
conda create --name <env_name> python=3.11
conda activate <env_name>
```

2. Install the repository's environment

```
cd rsa_square
pip install -e .
```

**Note:** If you prefer using `uv` over `conda`, I've successfully tested the following alternative workflow:

```
# Install python 3.11
uv python install 3.11
# Create a venv
uv venv --python 3.11 .venv
# Activate it
source .venv/bin/activate
# Install the project
uv pip install -e .
# Run scripts with python! e.g.
python run.py --config-path=configs --config-name=generate_alt_utterances gen_llm="meta-llama/Llama-3.1-8B" output_dir=data/prag_mega_plus_exps/llama_8b 
```

**Replicating our results:** For instructions about replicating our results, see the documentation in `src/README.md`.

# Citation

If you use the PragMega+ dataset or any of the ideas from our paper, please cite us:

```
@inproceedings{spinoso-di-piano-etal-2025-rsa,
    title = "(RSA)²: A Rhetorical-Strategy-Aware Rational Speech Act Framework for Figurative Language Understanding",
    author = "Spinoso-Di Piano, Cesare and Austin, David Eric and Piantanida, Pablo and Cheung, Jackie CK",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-long.1019/",
    doi = "10.18653/v1/2025.acl-long.1019",
    pages = "20898--20938",
    ISBN = "979-8-89176-251-0",
}
```