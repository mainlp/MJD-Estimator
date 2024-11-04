# MJD-Estimator
Implementation of the EMNLP 2024 paper from the MaiNLP Lab - ***"Seeing the Big through the Small": Can LLMs Approximate Human Judgment Distributions on NLI from a Few Explanations?*** ([paper](https://arxiv.org/pdf/2406.17600))

This repository contains the generator, evaluator and fine-tuning implementation for Model Judgment Distribution (MJDs) extracted from LLMs. We take Llama 3 as an example.

[**MJD-generator**](https://github.com/mainlp/MJD-Estimator/tree/main/MJD-generator): including the method to extract MJDs by first-token-probability (Section 3 in paper), the evaluation for distribution comparison (Section 4.3 in paper), and the code for ternary visualization (Section 6.1 in paper).

[**MJD-fine-tuning**](https://github.com/mainlp/MJD-Estimator/tree/main/MJD-fine-tuning): including the fine-tuning implementation for fine-tuning comparison (Section 4.4 in paper).


## Overall Structure
![Image text](https://github.com/mainlp/MJD-Estimator/blob/main/Overall_structure_EMNLP24.png)


## Citation
If you use this code, please cite the paper below:

["Seeing the Big through the Small": Can LLMs Approximate Human Judgment Distributions on NLI from a Few Explanations?](https://arxiv.org/pdf/2406.17600)

```
@article{DBLP:journals/corr/abs-2406-17600,
  author       = {Beiduo Chen and
                  Xinpeng Wang and
                  Siyao Peng and
                  Robert Litschko and
                  Anna Korhonen and
                  Barbara Plank},
  title        = {"Seeing the Big through the Small": Can LLMs Approximate
                  Human Judgment Distributions on {NLI} from a Few Explanations?},
  journal      = {CoRR},
  volume       = {abs/2406.17600},
  year         = {2024},
  url          = {https://doi.org/10.48550/arXiv.2406.17600},
  doi          = {10.48550/ARXIV.2406.17600},
  eprinttype    = {arXiv},
  eprint       = {2406.17600},
  timestamp    = {Mon, 22 Jul 2024 14:28:28 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2406-17600.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```


## Getting Started

### Setting up the code environment

```
$ conda env create -f <...>.yaml
```
notice that there are two conda environments for generator and fine-tuning.


### Datasets

The datasets for our experiments are from ChaosNLI and VariErrNLI.

ChaosNLI: NLI dataset with human judgment distributions (HJDs) from 100 crowd-workers. ([paper](https://arxiv.org/abs/2010.03532), [data](https://github.com/easonnie/ChaosNLI))

VariErrNLI: NLI dataset annotated with explanations by 4 linguistic experts. ([paper](https://aclanthology.org/2024.acl-long.123.pdf), [data](https://github.com/mainlp/VariErr-NLI))

We did a pre-process to extract the target filtered dataset as [NLI_explanations.json](https://github.com/mainlp/MJD-Estimator/blob/main/MJD-generator/NLI_explanations.json), which contains 341 overlapped NLI instances with 4 explanations for each.


### Running

**1. Move into the folder of module you chose**

`cd MJD-generator` or `cd MJD-fine-tuning`

#### Before you running any file, you need to modify the arguments to your own paths or hyper-parameters at first.

**2. Generation**

Generate the MJDs from Llama 3. The Llama 3 model is from [HuggingFace](https://huggingface.co/meta-llama) 

`ipython MJD-generator.ipynb`

**3. Evaluation**

Evaluate the MJDs with HJDs on distribution comparison metrics, and visualize the ternary plots.

`ipython MJD-evaluate.ipynb`

**3. Fine-Tuning**

Fine-tuning from a pretrained NLI model.

```
cd MJD-fine-tuning
bash train.sh
```


## License 
The code under this repository is licensed under the [Apache 2.0 License](https://github.com/mainlp/MJD-Estimator/blob/main/LICENSE).
