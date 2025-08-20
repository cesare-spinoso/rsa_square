# (RSA)²: A Rhetorical-Strategy-Aware Rational Speech Act Framework for Figurative Language Understanding

Welcome to this code repository! You will find the code and datasets used in the paper [(RSA)²: A Rhetorical-Strategy-Aware Rational Speech Act Framework for Figurative Language Understanding](https://aclanthology.org/2025.acl-long.1019/) (Spinoso-Di Piano et al., ACL 2025) presented in Vienna at ACL 2025.

# Datasets

**Experimental data and results:** We have uploaded all of our experimental datasets, results and analyses to this repository under the `data/` folder. We will provide instructions below on how to use the datasets in the `data/` folder to replicate our experiments and results.

**PragMega+ Hugging Face Dataset:** We have also created a Hugging Face 🤗 Hub dataset for our new **PragMega+** irony understanding dataset at [cesare-spinoso/PragMegaPlus](https://huggingface.co/datasets/cesare-spinoso/PragMegaPlus). To use it, you can use the following `python` snippet:

```
from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("cesare-spinoso/PragMegaPlus")
```

# Code

**File structure:**

```
|-
```

**Repository installation:** If you want to run our code, follow the steps below which install the repository.

0. Install the repository. We use [Git-LFS](https://git-lfs.com/) to store our experimental data and results. As a result, when you issue the `git clone` command, it **will not automatically download the experimental data and results**. To do so, you need to use `git clone` and `git lfs` together:

```
git clone https://github.com/cesare-spinoso/rsa_square.git
git lfs pull -I "data/**"
```

where the second line pulls the files from the LFS storage into the `data/` folder.

1. Create and activate a `conda` environment using `python 3.11`

```
conda create --name <env_name> python=3.11
conda activate <env_name>
```

2. Install the repository's environment

```
pip install -e .
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