# PEARL+
Source code and data sets for the PEARL+: An LLM-Enhanced Biterm Topic Modeling Framework for Low-Resource Personal Attribute Prediction from Conversations

# Requirements

Before running, you need to create an [unsloth]: https://github.com/unslothai/unsloth environment

In addition, 

# Data sets

We provide both the data sets (profession, hobby, 20News) in the folder ```data/```

# Quick start

```
bash start.sh
```

###### Parameter setting in start.sh

- ```gpu``` ➡ GPU to use; refer to nvidia-smi.
- ```dataset``` ➡ Data set name.
- ```model``` ➡  LLM model name: In this project, a **local** large language model is used and the path needs to be adjusted.
- ```K``` ➡ S, the number of selected words used by the LLM to synthesize texts.
- ```gen_len``` ➡ L, the length of texts synthesized by the LLM.
- ```part_of_dataset``` ➡ Synthesized quantity: the ratio of the number of synthesized texts to that of the original dataset.
- ```aaai_num``` ➡ Best BTM iterate times.
- ```num_keywords``` ➡ K, the number of keywords for each utterance.
- ```niter``` ➡ T, the number of iterations for the Gibbs sampling process.
