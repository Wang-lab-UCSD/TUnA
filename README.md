# TUnA: Transformer-based Uncertainty Aware model for PPI Prediction


## Introduction
This repository contains the data, code, and pretrained-models required to reproduce/use TUnA for PPI prediction.

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
#### Cross-species Dataset
```console
$ python3 process_xspecies.py 
```
#### Bernett Dataset
First we will construct the train/val/test datasets and embed using ESM-2 for the raw Bernett dataset. Once you are in the TUnA directory
```console
$ python3 process_bernett.py 
```
NOTE: We separate proteins into less than 1500 and greater than 1500. We use the GPU for sequences less than 1500 but use the CPU for sequences greater than 1500. Some sequences greater than 5000 require up to 300Gb of memory for embedding. We provide a pt file that contains these embeddings for download. Please use combine_dictionary.py to combine the two protein dictionaries into one. 
___
### Training, evaluation, and prediction

#### To train from scratch, head to results/ and choose the dataset/model you wish to train. Then, run:
```console
$ python3 main.py 
```
Hyperparameters and other options can be controlled using the config.yaml file. Please make sure the directories to the train/val/test dictionary and interaction files are correct. Every epoch, the performance on the validation set will be logged in output/results.txt


After training, to evaluate the model:
```console
$ python3 evaluate.py 
```
#### Using pre-trained models:
First, download the pretrained models you wish from: https://huggingface.co/datasets/yk0/TUnA_data/tree/main
Then, place the model file in the results/dataset/model/output directory. For example: place the bernett-TUnA model in the results/bernett/TUnA/output

Then run the inference:
```console
$ cd results/bernett/TUnA # navigate to the model you wish to use. The pretrained model needs to be placed in output/
$ python3 inference.py 
```
### Making predictions for any PPIs of interest
First, collect the proteins you wish to predict the interactions for. 
Format the first file such that there are two columns: Protein A, Protein B (Protein interaction file)
Format the second column such that it contains two columns: Protein, amino acid sequence (Protein dictionary file)


This will generate a predictions.tsv file that contains the columns: Protein A, Protein B, Predicted Interaction (0/1), and Uncertainty
Predictions with lower uncertainty will be more reliable than predictions with higher uncertainty. Based on our observations, 0.2 is a good upper-bound for the threshold, but depending on how risk-averse/risk-willing you are, you can decide which threshold is appropriate for you.
