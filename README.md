# MJD-Estimator & Model-Explanation
### **§1** -- ***"Seeing the Big through the Small": Can LLMs Approximate Human Judgment Distributions on NLI from a Few Explanations?*** ([paper](https://aclanthology.org/2024.findings-emnlp.842/)) @ EMNLP 2024


### **§2**  -- ***A Rose by Any Other Name: LLM-Generated Explanations Are Good Proxies for Human Explanations to Collect Label Distributions on NLI*** ([paper](https://aclanthology.org/2025.findings-acl.562/)) @ ACL 2025


This repository contains the generator, evaluator and fine-tuning implementation for Model Judgment Distribution (MJDs) extracted from LLMs. We take Llama 3 as an example.

[**MJD-generator**](https://github.com/mainlp/MJD-Estimator/tree/main/MJD-generator): including the method to extract MJDs by first-token-probability (Section 3 in **§1**), the evaluation for distribution comparison (Section 4.3 in **§1**), and the code for ternary visualization (Section 6.1 in **§1**).

[**MJD-fine-tuning**](https://github.com/mainlp/MJD-Estimator/tree/main/MJD-fine-tuning): including the fine-tuning implementation for fine-tuning comparison (Section 4.4 in **§1**).

[**Model-Explanation**](https://github.com/mainlp/MJD-Estimator/tree/main/Model-Explanation): including the LLM-generated and human-validated explanations for **§2**.



## Overall Structure
![Image text](https://github.com/mainlp/MJD-Estimator/blob/main/Overall_structure_EMNLP24.png)


## Citation
If you use this code&data, please cite the papers below:

["Seeing the Big through the Small": Can LLMs Approximate Human Judgment Distributions on NLI from a Few Explanations?](https://aclanthology.org/2024.findings-emnlp.842/)

```
@inproceedings{chen-etal-2024-seeing,
    title = "{``}Seeing the Big through the Small{''}: Can {LLM}s Approximate Human Judgment Distributions on {NLI} from a Few Explanations?",
    author = "Chen, Beiduo  and
      Wang, Xinpeng  and
      Peng, Siyao  and
      Litschko, Robert  and
      Korhonen, Anna  and
      Plank, Barbara",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2024",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-emnlp.842",
    pages = "14396--14419",
    abstract = "Human label variation (HLV) is a valuable source of information that arises when multiple human annotators provide different labels for valid reasons. In Natural Language Inference (NLI) earlier approaches to capturing HLV involve either collecting annotations from many crowd workers to represent human judgment distribution (HJD) or use expert linguists to provide detailed explanations for their chosen labels. While the former method provides denser HJD information, obtaining it is resource-intensive. In contrast, the latter offers richer textual information but it is challenging to scale up to many human judges. Besides, large language models (LLMs) are increasingly used as evaluators ({``}LLM judges{''}) but with mixed results, and few works aim to study HJDs. This study proposes to exploit LLMs to approximate HJDs using a small number of expert labels and explanations. Our experiments show that a few explanations significantly improve LLMs{'} ability to approximate HJDs with and without explicit labels, thereby providing a solution to scale up annotations for HJD. However, fine-tuning smaller soft-label aware models with the LLM-generated model judgment distributions (MJDs) presents partially inconsistent results: while similar in distance, their resulting fine-tuned models and visualized distributions differ substantially. We show the importance of complementing instance-level distance measures with a global-level shape metric and visualization to more effectively evaluate MJDs against human judgment distributions.",
}

```


[A Rose by Any Other Name: LLM-Generated Explanations Are Good Proxies for Human Explanations to Collect Label Distributions on NLI](https://aclanthology.org/2025.findings-acl.562/)



```
@inproceedings{chen-etal-2025-rose,
    title = "A Rose by Any Other Name: {LLM}-Generated Explanations Are Good Proxies for Human Explanations to Collect Label Distributions on {NLI}",
    author = "Chen, Beiduo  and
      Peng, Siyao  and
      Korhonen, Anna  and
      Plank, Barbara",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2025",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-acl.562/",
    pages = "10777--10802",
    ISBN = "979-8-89176-256-5",
    abstract = "Disagreement in human labeling is ubiquitous, and can be captured in human judgment distributions (HJDs). Recent research has shown that explanations provide valuable information for understanding human label variation (HLV) and large language models (LLMs) can approximate HJD from a few human-provided label-explanation pairs. However, collecting explanations for every label is still time-consuming. This paper examines whether LLMs can be used to replace humans in generating explanations for approximating HJD. Specifically, we use LLMs as annotators to generate model explanations for a few given human labels. We test ways to obtain and combine these label-explanations with the goal to approximate human judgment distributions. We further compare the resulting human with model-generated explanations, and test automatic and human explanation selection. Our experiments show that LLM explanations are promising for NLI: to estimate HJDs, generated explanations yield comparable results to human{'}s when provided with human labels. Importantly, our results generalize from datasets with human explanations to i) datasets where they are not available and ii) challenging out-of-distribution test sets."
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

```
cd MJD-generator
```

or

```
cd MJD-fine-tuning
```

#### Before you running any file, you need to modify the arguments to your own paths or hyper-parameters at first.

**2. Generation**

Generate the MJDs from Llama 3. The Llama 3 model is from [HuggingFace](https://huggingface.co/meta-llama) 

```
ipython MJD-generator.ipynb
```

**3. Evaluation**

Evaluate the MJDs with HJDs on distribution comparison metrics, and visualize the ternary plots.

```
ipython MJD-evaluate.ipynb
```
**4. Fine-Tuning**

Fine-tuning from a pretrained NLI model.

```
cd MJD-fine-tuning
bash train.sh
```


## License 
The code under this repository is licensed under the [Apache 2.0 License](https://github.com/mainlp/MJD-Estimator/blob/main/LICENSE).
