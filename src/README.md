# Replicating our results

We have documented how to replicate the experiments discussed in the main part of our paper. This document guides you on how to run those experiments and generate the tables and figures related to them. Some additional notebooks are created for some of the results found in the appendix.

**Q: I just want to look at your experimental data and results.** 

**A:** We have uploaded all of our experimental data and results to this repository in the `data/` folder.

**Q: I just want to use PragMega+ for my own research.**

**A:** We have uploaded the PragMega+ to the Hugging Face Dataset repository [cesare-spinoso/PragMegaPlus](https://huggingface.co/datasets/cesare-spinoso/PragMegaPlus). You can download it using the `dataset` `python` package  using the following code snippet:

```
from datasets import load_dataset

dataset = load_dataset("cesare-spinoso/rsa_square_data")
```

# Non-Literal Number Expressions Experiments

This section includes instructions on reproducing results from our non-literal number expressions experiment including both the affect-aware RSA and (RSA)^2 experiments. The metrics and plots in the paper are generated with the python script `nonliteral_numbers_exp.py`. The data used in this script is found in `data/nonliteral_number_exps`.

To reproduce all our results for this experiment, run the following python script:

```
python nonliteral_numbers_exp.py
```

**Compute**: You can run this script on a CPU!

**Expected output**: The following files should be created upon running this command:

```
nonliteral_number_exps/
├── meaning_probabilities_merged.csv # Human, affect-aware and (RSA)^2 meaning probabilities
├── metrics.csv # Metrics for the affect-aware and (RSA)^2 listeners
# Marginalized (RSA)^2 listener probabilities
├── pl0_marg_df.csv
├── pl1_marg_df.csv
# Listener probability bar plots for the different object price utterances
├── state_probabilities_context=electric kettle_splitidx=0.png
├── state_probabilities_context=electric kettle_splitidx=1.png
├── state_probabilities_context=laptop_splitidx=0.png
├── state_probabilities_context=laptop_splitidx=1.png
├── state_probabilities_context=watch_splitidx=0.png
└── state_probabilities_context=watch_splitidx=1.png
 ```


### Ironic Weather Utterances Experiments

This section includes instructions on reproducing results from our ironic weather utterances experiment including both the affect-aware RSA and (RSA)^2 experiments. The metrics and plots in the paper are generated with the python script described below.

If you want to reproduce our training run for the ironic weather utterances experiment use the following:

```
python weather_utterances_exp.py --config-path=configs --config-name=weather_utterances_exp train_network=true output_dir=<name_of_your_output_dir>
```

You can play around with a few of the hyperparameters such as the learning rate, weight decay and number of epochs (the model with the lowest validation loss is chosen) in the config for this training script found in `src/configs/weather_utterances_exp.yaml`.

If you simply want to re-run inference (no training from scratch) with the rhetorical strategy network which we trained, then run the following
```
python weather_utterances_exp.py --config-path=configs --config-name=weather_utterances_exp output_dir=<name_of_your_output_dir>
```

**Compute:** If you are training the rhetorical function network from scratch, then a GPU (albeit a small one) is likely necessary. If you are only running the inference version of the script, then a CPU should be enough.

**Expected output:** The following files should be created upon running this command:

```
ironic_weather_utterances_exps/
# (RSA)^2 training and inference artifacts (without any postprocessing)
├── model.pth # New or saved rhetorical function network weights
├── test_predictions.csv
├── train_predictions.csv
├── trained_rf_values.csv
# Listener distributions
├── human_dist_train.csv # Human meaning distributions on the train set
├── human_dist_test.csv # Human meaning distributions on the test set
├── affect_aware_rsa.csv # Affect-aware RSA predictions (they are the same for the train and test set)
├── rsa_two_train.csv # (RSA)^2 meaning distributions on the train set
├── rsa_two_test.csv # (RSA)^2 meaning distributions on the test set
# Listener distribution metrics
├── metrics_test.csv # Test-time metrics
├── metrics_train.csv # Train-time metrics
# Figure reproducing the affect-aware RSA plot from Kao
├── affect_aware_rsa_l1.png
# Listener probability bar plots for each weather-context/utterance pair
├── bar_plot_amazing_1.png
├── bar_plot_amazing_2.png
├── bar_plot_amazing_3.png
├── bar_plot_ok_4.png
├── bar_plot_ok_5.png
├── bar_plot_ok_6.png
├── bar_plot_terrible_7.png
├── bar_plot_terrible_8.png
└── bar_plot_terrible_9.png

```

**Note about specifying paths:** When configuring the `output_dir` path variable (or any path variable in this repository), make sure to write a path which is *relative* to this repository. For example, if you want the results to end up in `path_to_this_repo/results/new_run`, then you should set the `output_dir` to `results/new_run`. 

### PragMega+ Experiments

We show the steps required for reproducing our LLM RSA + LLM (RSA)^2 experiment results on the PragMega+ dataset which we release on [Huggingface]() as part of this publication. [TODO: Add HuggingFace links here] The validation set we use can be found in `src/data/prag_mega/ScenariosOriginalAndFlipped.v2.csv` and the test set we use can be found in `src/data/prag_mega/ScenariosCombinedDiversified.csv`. We have documented how we created this dataset in `src/data/prag_mega/README_camera_ready.md`.

We show the steps for reproducing our results on the test set below(we mostly used the validation set for prompt engineering).

#### Step 0: Repository setup

We first need to install this GitHub directory, its corresponding python environment and its dataset.
    - Install the directory.
    - Install the python environment with ...
    - Download the datasets by running ... [add a make script to download from huggingface automatically]

#### Step 1: Alternative utterance generation

**Generate alternative utterances:** We generate 50 alternative utterances for each scenario so that we can use them for the RSA and (RSA)^2 Bayesian posterior update. The alternative utterances are generated by providing a (base) LLM with a scenario that contains several characters and by asking it to complete the utterance of the character who is about to speak (e.g., `[scenario] John says "`). To generate these alternative utterances for the test set you can use the command below:

```
python run.py --config-path=configs/camera_ready --config-name=generate_alt_utterances gen_llm="meta-llama/Llama-3.1-8B" output_dir=camera_ready_outputs/prag_mega_plus_exps/llama_8b
```

**Compute**: You can run this script using a single Quadro RTX 8000.

**Expected output:** A time-stamped directory will automatically be created under the `output_dir` in the above command. This time-stamped directory will contain the config used to generate the dataset (YAML file) and the alternative utterances (`alternative_utterances.jsonl`). The alternative utterance file will have the following structure:

```
{"label": "IV", "id": 1, "alt_id": 1, "alt_utt": "\"You should clean your room. It's not fair for you to make me clean up.\""}
{"label": "IV", "id": 1, "alt_id": 2, "alt_utt": "\"Kids, come and help me clean your room!\""}
{"label": "IV", "id": 1, "alt_id": 3, "alt_utt": "\"Who put this mess here?! Get up and clean your room right now!\""}
{"label": "IV", "id": 1, "alt_id": 4, "alt_utt": "\"Clean up your room right now or there will be consequences!\""}
{"label": "IV", "id": 1, "alt_id": 5, "alt_utt": "\"Clean your room or else.\""}
```

where `id` is the scenario id (ranges from 1 to 50 where the first 25 are the ironic scenarios and the last 25 are the corresponding literal scenarios) and the `alt_id` is the alternative id (ranges from 1 to 50, duplicates are not removed).

A seed has been set in this config for reproducibility, but in case there are any discrepancies we have also included the alternative utterances and the corresponding generation config in `camera_ready_outputs/prag_mega_plus_exps/llama_8b` which we used for our experiments. These should have been downloaded automatically from HuggingFace upon repository installation.

**Extra Note**: We identified alternative utterance generation to be the major source of difficulty for using RSA and (RSA)^2 with LLMs. Most alternative utterances are paraphrases of one another, and thus they do not provide the pragmatic listener with useful alternatives to enable the bayesian posterior update to *narrow* the probability distribution on the pragmatic meaning.

#### Step 2: Generate prompt-based LLM probabilities

**Generate prompt-based LLM probabilities:** We use an instruction-tuned LLM to compute the meaning prior `PLLM(m|c)`, the meaning posterior with and without the rhetorical strategy random variable in the conditional `PLLM(m|c,u)` and `PLLM(m|c,u,r)` and the rhetorical strategy posterior `PLLM(r|c,u)`. The prompt templates are based on different versions we tried on the validation set. You can run the script to compute these probabilities with the command below.

```
python run.py --config-path=configs/camera_ready --config-name=generate_llm_prompt_probs prob_llm="mistralai/Mistral-7B-Instruct-v0.3" alternative_utterances_path=<path_to_alternatives_created_in_step_1>
```

If you would rather not generate the alternatives in step 1 yourself, then you can use the path to the alternatives we created `camera_ready_outputs/prag_mega_plus_exps/llama_8b/alternative_utterances.jsonl`, making the command:

```
python run.py --config-path=configs/camera_ready --config-name=generate_llm_prompt_probs prob_llm="mistralai/Mistral-7B-Instruct-v0.3" alternative_utterances_path=camera_ready_outputs/prag_mega_plus_exps/llama_8b/alternative_utterances.jsonl
```

**Compute**: You can run this script using a single Quadro RTX 8000.

**Expected output:** Like step 1., this will also create a time-stamped directory containing the output probability file (`llm_prompt_probs.jsonl`) and the config used to generate it (YAML file). For each scenario, the LLM probability file will contain metadata (e.g., hyperparameter values) as well as scenario specific data (e.g., the original utterance produced by the character) and probabilities under the key "PLLM" which will have the following structure:

```
{
    "id": "Scenario id",
    "meanings": "List of meaning interpretations",
    "scenario_wo_question": "Scenario without the question `What did the character want to convey?`",
    "original_utterance": "Utterance produced by the character",
    "context_w_something": "Scenario where the character's utterance is replaced with `The character said something.` This is used to compute the prior probability.",
    "PLLM": {
    {
      "PLLM(m|c)": {
        "type": "dict",
        "description": "Probabilities of meanings given context only.",
        "keys": "tuple containing the `scenario_wo_question` string",
        "values": {
          "type": "dict",
          "description": "Mapping from meaning text to probability score.",
          "value_type": "float (0.0 - 1.0)"
        }
      },
      "PLLM(m|c,u)": {
        "type": "dict",
        "description": "Probabilities of meanings given both `scenario_wo_question` and utterance. Conditional distribution for each utterance (original and alternative) is placed here. To index into the distribution of the original utterance, use `original_utterance`.",
        "keys": "tuple of (`scenario_wo_question` string, utterance string)",
        "values": {
          "type": "dict",
          "description": "Mapping from meaning text to probability score.",
          "value_type": "float (0.0 - 1.0)"
        }
      },
      "PLLM(m|c,u,r)": {
        "type": "dict",
        "description": "Probabilities of meanings given context, utterance, and rhetorical strategy.",
        "keys": "tuple of (`scenario_wo_question` string, utterance string, rhetorical strategy string)",
        "values": {
          "type": "dict",
          "description": "Mapping from meaning text to probability score.",
          "value_type": "float (0.0 - 1.0)"
        }
      },
      "PLLM(r|c,u)": {
        "type": "dict",
        "description": "Probabilities of rhetorical strategies given context and utterance.",
        "keys": "tuple of (`scenario_wo_question` string, utterance string)",
        "values": {
          "type": "dict",
          "description": "Mapping from rhetorical strategy label (e.g., 'Ironic', 'Sincere') to probability score.",
          "value_type": "float (0.0 - 1.0)"
        }
      }
    }
  }
}
```

**Caching:** We have implemented a caching mechanism where we cache the string used for eliciting a given probability and re-use the cached probability if it re-appears (e.g., if a session crashes). The cache file for the probabilities elicited using "mistralai/Mistral-7B-Instruct-v0.3" should also have been downloaded from HuggingFace and placed in `camera_ready_outputs/prag_mega_plus_exps/mistral_7b_instruct`.

If you don't do anything, the script will automatically look for the cache, find it full and use the cached probabilities to generate the JSONL probability file. If you want to re-run everything from scratch then specify a new path for the cache file (e.g., a new empty directory) in the config file of the above command (the key for the cache file in the config is `cache_dir`). **WARNING:** Re-running everything from scratch will likely require a half a day to a day's worth of compute.

#### Step 3: Generate utterance probabilities

**Generate utterance probabilities:** The final script to run *before* we can use RSA and (RSA)^2 is the utterance probabilities PLLM(u|c) used in the speaker equation. These probabilities are computed by taking the conditional log-likelihood of each utterance given the context in the scenario, averaging it over the number of tokens, taking the exp and normalizing to recover a probability.

These probabilities are generated separately from Step 2 since they do not require a prompt template. We could have done this in Step 1, but we chose to keep the generation of alternatives and their probability computation separate. We compute these probabilities using the following command:

```
python run.py --config-path=configs/camera_ready --config-name=generate_utterance_probs alternative_utterances_path=camera_ready_outputs/prag_mega_plus_exps/llama_8b/alternative_utterances.jsonl alternative_utterances_path=<path_to_alternatives_created_in_step_1> llm_prompt_probs_path=<path_to_llm_prompt_probs_created_in_step_2> gen_llm="meta-llama/Llama-3.1-8B"
```

If you would rather not generate 1. and 2. from scratch, you can use the alternative utterances and the LLM prompt probs we used in our experiments (paths in the command below). The command would then become:

```
python run.py --config-path=configs/camera_ready --config-name=generate_utterance_probs alternative_utterances_path=camera_ready_outputs/prag_mega_plus_exps/llama_8b/alternative_utterances.jsonl alternative_utterances_path="camera_ready_outputs/prag_mega_plus_exps/llama_8b/alternative_utterances.jsonl" llm_prompt_probs_path="camera_ready_outputs/prag_mega_plus_exps/mistral_7b_instruct/llm_prompt_probs.jsonl" gen_llm="meta-llama/Llama-3.1-8B"
```

**Compute**: You can run this script using a single Quadro RTX 8000.

**Expected output:** Like in step 1. and step 2., this will also create a time-stamped directory containing the utterance probability file (`llm_utterance_probs.jsonl`) and the config used to generate it (YAML file). The script also creates a merged JSONL probability file called `llm_probs.jsonl`. The utterance probability JSONL file will have the following structure:

```
{
    "id": "Scenario id",
    "PLLM": {
        "PLLM(u|c)": {
        "type": "dict",
        "description": "Probabilities of utterance.",
        "keys": "tuple of (`scenario_wo_question` string, utterance string)",
        "values": {
                "type": "dict",
                "description": "Mapping from rhetorical strategy label (e.g., 'Ironic', 'Sincere') to probability score.",
                "value_type": "float (0.0 - 1.0)"
            }
        }
    }
}
```

and the merged probability file will have the same structure as the file in 2. with the additional PLLM(u|c) field.

#### Step 4: Running the RSA & (RSA)^2 experiments

**Running the RSA experiments:** All of the different versions of LLM + RSA listeners (including the clustering algorithm found in the appendix) are implemented in `src/rsa.py`. A config to run the LLM-RSA, LLM-(RSA)^2 with P(r|c,u), LLM-(RSA)^2 with the indicator function instead of P(r|c,u) and the RSA clustering listeners is found in `configs/rsa.yaml`. The command for running a script for these 4 versions (the first three being the runs used for main LLM results in the paper) is the following:

```
python rsa.py --config-path=configs/camera_ready --config-name=rsa alt_utterances_path=<alternative_utterance_path_from_step_1> llm_probs_path=<merged_llm_prob_path_from_step_3>
```

If you haven't run any of the previous steps and just care about running the RSA scripts, then you can use the JSONL files we provide (which we used in the for the paper as well) using the following command:

```
python rsa.py --config-path=configs/camera_ready --config-name=rsa alt_utterances_path="camera_ready_outputs/prag_mega_plus_exps/llama_8b/alternative_utterances.jsonl" llm_probs_path="camera_ready_outputs/prag_mega_plus_exps/llm_probs.jsonl"
```

**Compute:** A CPU should be enough for the LLM-RSA and LLM-(RSA)^2 listeners. A GPU (albeit a fairly small one) is recommended for the clustering algorithm as we use an embedding model to compute vector representations of the utterances.

**Expected output:** A time-stamped directory will be created within the `camera_ready_outputs/prag_mega_plus_exps/rsa_runs` directory which will have the following structure:

```
.
├── config.yaml # Config for that RSA run
# Meaning probabilities for the L0 listeners
├── L0_llm_rsa.jsonl
├── L0_rsc_rsa.jsonl
├── L0_rsa_two_indicator_r_c_u.jsonl
├── L0_rsa_two_p_r_c_u.jsonl
# Meaning probabilities for the L1 listeners
├── L1_llm_rsa.jsonl
├── L1_rsc_rsa.jsonl
├── L1_rsa_two_indicator_r_c_u.jsonl
├── L1_rsa_two_p_r_c_u.jsonl
# Utterance probabilities for the S1 speakers
├── S1_llm_rsa.jsonl
├── S1_rsc_rsa.jsonl
├── S1_rsa_two_indicator_r_c_u.jsonl
└── S1_rsa_two_p_r_c_u.jsonl
```

The structure of the listener JSONL files (i.e., any file of the form `{L0,L1}_{llm_rsa,rsc_rsa,rsa_two_indicator_r_c_u,rsa_two_p_r_c_u}.jsonl`) is the same as the merged LLM probability JSONL file (`llm_probs.jsonl`) except that it contains an extra key for the corresponding listener probabilities. For instance, for the L1 listener in the `L1_llm_rsa.jsonl` file, the additional dict structure would be the following:
```
{
    "L1": {
        "PL1_llm_rsa(m|c,u)": {
            "type": "dict",
            "description": "Meaning probabilities for the L1 LLM RSA listener.",
            "keys": "tuple of (`scenario_wo_question` string, utterance string)",
            "values": {
                    "type": "dict",
                    "description": "Meaning probability for each meaning option. To get the probability of the original utterance, you need to index with the original utterance string found in the metadata.,
                    "value_type": "float (0.0 - 1.0)"
                }
        }
    }
}
```

**Extra note:** Feel free to play around with different hyperparameter settings in the `rsa.yaml` config file (e.g., changing the rationality parameter alpha, running on more than one RSA iteration, changing the prior to uniform, etc.). Running the script is fairly quick since all the RSA computations are vectorized. This is also why we chose to separate the LLM computation from the RSA one.

#### Step 5: Generating results, ablations and plots

**Generating ALL the experimental results:** To run all the experiments which are featured as results in the main section of our paper, we have created a script which runs different version of the `rsa.yaml` config automatically. This feature is especially useful for recreating the ablations we present in the ablation section of our paper. To run all the experiments needed to recreate all the results, plots and ablations in the main part of our paper, run the following command:

```
python run_all_rsa.py --config-path=configs/camera_ready/ --config-name=main_all_rsa_runs base_config_common_keys.alt_utterances_path=<alternative_utterance_path_from_step_1> base_config_common_keys.llm_probs_path=<merged_llm_prob_path_from_step_3> 
```

Or, if you're using the files we generated, then:

```
python run_all_rsa.py --config-path=configs/camera_ready/ --config-name=main_all_rsa_runs base_config_common_keys.alt_utterances_path=camera_ready_outputs/prag_mega_plus_exps/llama_8b/alternative_utterances.jsonl base_config_common_keys.llm_probs_path=camera_ready_outputs/prag_mega_plus_exps/llm_probs.jsonl
```

**Compute:** A CPU should be enough! We aren't using the clustering algorithm in this case, so no GPUs are needed. All the RSA operations are vectorized!

**Expected output:** In this case **no time-stamped directory is created** (so beware of overriding!). Instead, a directory will be created in `camera_ready_outputs/prag_mega_plus_exps/rsa_runs` which will follow the syntax: `gen_llm={gen_llm}_prob_llm={prob_llm}main_paper` where, in this case, `gen_llm=llama_8b` and `prob_llm=mistral_7b`, but if you used a different pair of LLMs, then you would need to specify this in the `main_all_rsa_runs.yaml` config file.

The generated directory will contain the different listener probabilities for the un-ablated as well as the ablated versions of the listeners. It will also contain a YAML log file of the paths to each listener probability results which will be used for creating the tables and plots in the next command:

```
├── gen_llm=llama_8b_prob_llm=mistral_7b_instructmain_paper_results_and_analysis.yaml
├── gen_llm=llama_8b_prob_llm=mistral_7b_instructmain_paper # Directory with listener probabilities
    ├── main_run # Un-ablated version
    ├── no_p_m_c # Ablation without the meaning prior in L1 (i.e., using uniform instead)
    ├── no_p_r_c_u # Ablation without P(r|c,u) in the marginalization (i.e., using uniform instead)
    └── no_p_u_c # Ablation without the utterance prior in S1 (i.e., using uniform instead)
```

**Generating the tables and plots:** To generate all the tables and plots found in the main paper, you can use the following script which uses `gen_llm=llama_8b_prob_llm=mistral_7b_instructmain_paper_results_and_analysis.yaml` created by the previous command. To run the command to generate the results and analysis in the main section of our paper, use the following command (this last one is very easy!):

```
python results_and_analysis.py --config-path=configs/camera_ready --config-name="gen_llm=llama_8b_prob_llm=mistral_7b_instructmain_paper_results_and_analysis"
```

**Compute:** CPU!

**Expected output:** The tables and plots used in the main section of our paper.

```
results_and_analysis/gen_llm=llama_8b_prob_llm=mistral_7b_instructmain_paper
# Average meaning probabilities (Table 2)
├── gen_llm=llama_8b_prob_llm=mistral_7b_instructmain_paper_average_probabilities_correct_incorrect.csv
├── gen_llm=llama_8b_prob_llm=mistral_7b_instructmain_paper_average_probabilities_table.tex
# Average meaning probabilities, grouped by ironic vs literal scenario
├── gen_llm=llama_8b_prob_llm=mistral_7b_instructmain_paper_average_probabilities_ironic.csv
├── gen_llm=llama_8b_prob_llm=mistral_7b_instructmain_paper_average_probabilities_literal.csv
# Meaning probabilities bar plot (Figure 4)
├── gen_llm=llama_8b_prob_llm=mistral_7b_instructmain_paper_listener_meaning_probabilities.pdf
# Ablation study (Table 3)
├── prior_ablations_table.csv
└── prior_ablations_table.tex
```

That's it for replicating the results in the main section of our paper. Keep reading for some appendix results!

#### Step 6 (Optional): Appendix results

Tables 5 and 6: Run the notebook `notebooks/appendix_table_5_and_6.ipynb`
Table 7: Run the notebook `notebooks/appendix_table_7.ipynb`. This notebook uses clustering runs which are already generated for simplicity though you can still look at their corresponding `config.yaml` file if you have a doubt about a hyperparameter setting.