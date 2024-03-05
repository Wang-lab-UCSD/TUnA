from data_processing.data_utils import *
from data_processing.embed import *
import os   
# This script will process the raw data and embed using ESM-2.  

"""First, process the raw Bernett data"""
def make_bernett_interactions(dataset, target_dir, fasta):
    negative_file = f"{DIR}{intra}_neg_rr.txt"
    positive_file = f"{DIR}{intra}_pos_rr.txt"
    interaction_file = f"{dataset}_interaction.tsv"
    dictionary_file = f"{dataset}_dictionary.tsv"
    output_interaction_path = os.path.join(target_dir, interaction_file)
    output_dictionary_path = os.path.join(target_dir, dictionary_file)
    process_and_combine_tsv(negative_file, positive_file, output_interaction_path, SEED)
    create_unique_proteins_with_sequences_file(output_interaction_path, fasta, output_dictionary_path)
    separate_files_based_on_sequence_length(dataset, MAX_LENGTH, target_dir)

MAX_LENGTH = 1500
DATASETS = ['Intra0', 'Intra1', 'Intra2']
DIR = "data/raw/bernett/"
FASTA_FILE = "data/raw/bernett/human_swissprot_oneliner.fasta"
TARGET_DIR = "data/processed/bernett/"
SEED = 47 # Used for shuffling the rows of the interaction file

os.makedirs(os.path.dirname(TARGET_DIR), exist_ok=True)
for intra in DATASETS:
    make_bernett_interactions(intra, TARGET_DIR, FASTA_FILE)

"""Second, embed the processed Bernett data with ESM-2"""
"""Please change the cuda device to the device that you are using"""
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
model, alphabet = load_model_and_alphabet(device)
datasets = ['Intra0_dictionary_1500_or_less','Intra1_dictionary_1500_or_less','Intra2_dictionary_1500_or_less']
for dataset in datasets:
    data_file = os.path.join(TARGET_DIR, f"{dataset}.tsv")
    with open(data_file, "r") as f:
        data_list = f.read().strip().split('\n')
    dir_input = os.path.join('data/embedded/bernett/', dataset)
    os.makedirs(dir_input, exist_ok=True)
    process_data_points(model, alphabet, data_list, dir_input, device=device)
