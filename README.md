# Modeling Structural Similarities between Documents for Coherence Assessment with Graph Convolutional Networks
Code for the ACL 2023 paper "Modeling Structural Similarities between Documents for Coherence Assessment with Graph Convolutional Networks"

## Requirement
Our working environment is Python 3.8. Before you run the code, please make sure you have installed all the required packages. You can achieve it by simply execute the shell as `sh requirements.sh`

## GCDC
To run experiments on GCDC, you should:
1. Put the raw corpora under the folder "data/dataset/raw/gcdc"
2. Convert raw data into json files via `python3 preprocessing.py`
3. Call the script. For example, you can `sh script/run_clinton.sh` to run experiments on gcdc_clinton.
