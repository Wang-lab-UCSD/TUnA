from data_processing.data_utils import *
from data_processing.embed import *
import os   
# This script will process the raw data and embed using ESM-2.  

"""First, process the raw x-species data"""
def make_xspecies_interactions(dataset, dir, target_dir, fasta_dir):
    species_name = dataset.split('_')[0]
    input_file = os.path.join(dir, f"{dataset}.tsv")
    interaction_file = f"{dataset}_interaction.tsv"
    dictionary_file = f"{dataset}_dictionary.tsv"
    output_interaction_path = os.path.join(target_dir, interaction_file)
    output_dictionary_path = os.path.join(target_dir, dictionary_file)
    shuffle_and_clean(input_file, output_interaction_path, SEED)
    fasta_location = os.path.join(fasta_dir, f"{species_name}.fasta")
    create_unique_proteins_with_sequences_file(output_interaction_path, fasta_location, output_dictionary_path)

DATASETS = ['human_train','human_test','mouse_test','fly_test','worm_test','yeast_test','ecoli_test']
DIR = "data/raw/xspecies/pairs/"
FASTA_FILE = "data/raw/xspecies/seqs/"
TARGET_DIR = "data/processed/xspecies/"
SEED = 47 # Used for shuffling the rows of the interaction file

os.makedirs(os.path.dirname(TARGET_DIR), exist_ok=True)
for species in DATASETS:
    make_xspecies_interactions(species, DIR, TARGET_DIR, FASTA_FILE)

"""Second, embed the processed x-species data with ESM-2"""
"""Please change the cuda device to the device that you are using"""
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
model, alphabet = load_model_and_alphabet(device)
datasets = ['human_train_dictionary','human_test_dictionary','mouse_test_dictionary','fly_test_dictionary','worm_test_dictionary','yeast_test_dictionary','ecoli_test_dictionary']
for dataset in datasets:
    data_file = os.path.join(TARGET_DIR, f"{dataset}.tsv")
    with open(data_file, "r") as f:
        data_list = f.read().strip().split('\n')
    dir_input = os.path.join('data/embedded/xspecies/', dataset)
    os.makedirs(dir_input, exist_ok=True)
    process_data_points(model, alphabet, data_list, dir_input, device=device)
