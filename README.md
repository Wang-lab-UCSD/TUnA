# TUnA: Transformer-based Uncertainty Aware model for PPI Prediction


## Introduction
This repository contains the data and code required to reproduce results or run TUnA.

## Installation
```console
$ git clone https://github.com/Wang-lab-UCSD/TUnA
$ cd TUnA
$ conda env create --file environment.yml
$ conda activate tuna
```
NOTE: The torch packages in environment.yml may need to be edited depending on which CUDA you are using: https://pytorch.org/get-started/previous-versions/ 

## Usage
### Data processing
The embedding step may take some time.
#### Cross-species Dataset
```console
$ python3 process_xspecies.py 
```
#### Bernett Dataset
```console
$ python3 process_bernett.py 
```
NOTE:The embedded Bernett data can be downloaded here: https://huggingface.co/yk0/TUnA_embeddings/tree/main. Please place the three folders in the data/embedded/bernett/ directory.
___
### Training and evaluation

#### To train from scratch, head to results/ and choose the dataset/model you wish to train. Then, run:
```console
$ python3 main.py 
```
Hyperparameters and other options can be controlled using the config.yaml file. Please make sure the directories to the train/val/test dictionary and interaction files are correct. Every epoch, the performance on the validation set will be logged in output/results.txt

#### Using pre-trained models:
First, download the pretrained models you wish from: [https://huggingface.co/datasets/yk0/TUnA_data/tree/main](https://huggingface.co/datasets/yk0/TUnA_models).
Then, place the model file in the results/dataset/model/output directory. For example: place the bernett-TUnA model in the results/bernett/TUnA/output

Then to evaluate either the re-trained or pre-trained models on the test sets:
```console
$ cd results/bernett/TUnA # navigate to the model you wish to use. The pretrained model needs to be placed in output/
$ python3 inference.py 
```
