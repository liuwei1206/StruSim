# Modeling Structural Similarities between Documents for Coherence Assessment with Graph Convolutional Networks
Code for the ACL 2023 paper "[Modeling Structural Similarities between Documents for Coherence Assessment with Graph Convolutional Networks](https://arxiv.org/pdf/2306.06472.pdf)"

If any questions, please contact the email: willie1206@163.com

## 1 Requirement
Our working environment is Python 3.8. Before you run the code, please make sure you have installed all the required packages. You can achieve it by simply execute the shell as `sh requirements.sh`

Then you should prepare embedding, xlnet, and stanza:
1. Download embedding from [here](https://nlp.stanford.edu/data/glove.840B.300d.zip) and put it under the folder "data/embedding".
2. Download xlnet-base_cased from [here](https://huggingface.co/xlnet-base-cased/tree/main) and put it under the folder "data/pretrained_models".
3. Download stanza resource via `python3 preprocessing.py` and put it under the folder "data/stanza".

## 2 GCDC
To run experiments on GCDC, you should:
1. Put the raw corpora under the folder "data/dataset/raw/gcdc"
2. Convert raw data into json files via `python3 preprocessing.py`
3. Call the script. For example, you can `sh script/run_clinton.sh` to run experiments on gcdc_clinton.

## 3 GCDC
To run experiments on Toefl, you should:
1. Put the raw corpora under the folder "data/dataset/raw/toefl"
2. Convert raw data into json files via `python3 preprocessing.py`
3. Call the script. For example, you can `sh script/run_toefl1.sh` to run experiments on the prompt 1 of toefl corpus.

## 4 Citation
```
@misc{liu2023modeling,
      title={Modeling Structural Similarities between Documents for Coherence Assessment with Graph Convolutional Networks}, 
      author={Wei Liu and Xiyan Fu and Michael Strube},
      year={2023},
      eprint={2306.06472},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
