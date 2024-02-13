import csv
import random
import pandas as pd

def write_interaction_data(file_path, data):
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        for row in data:
            # Check if the row has at least 3 columns
            if len(row) >= 3:
                # Try converting the third column to an integer
                try:
                    row[2] = int(float(row[2]))
                except ValueError:
                    # Handle the case where conversion is not possible
                    print(f"Warning: Could not convert {row[2]} to int. Keeping the original value.")
            writer.writerow(row)

def process_and_combine_tsv(negative_file, positive_file, output_file, seed=47):
    random.seed(seed)
    def read_interaction_data(file_path, label):
        with open(file_path, 'r', newline='') as file:
            return [row.split() + [label] for row in file]
    # Combine negative and positive interaction data
    negative_data = read_interaction_data(negative_file, '0')
    positive_data = read_interaction_data(positive_file, '1')
    combined_data = negative_data + positive_data
    # Shuffle the combined data
    random.shuffle(combined_data)

    write_interaction_data(output_file, combined_data)

def shuffle_and_clean(input_file, output_file, seed=47):
    random.seed(seed)
    def read_interaction_data(file_path,):
        with open(file_path, 'r', newline='') as file:
            return [row.split() for row in file]
    # Combine negative and positive interaction data
    data = read_interaction_data(input_file)
    # Shuffle the combined data
    random.shuffle(data)
    write_interaction_data(output_file, data)

def write_protein_sequences(file_path, proteins, sequences):
    with open(file_path, 'w', newline='') as file:
        for protein in proteins:
            sequence = sequences.get(protein, "Sequence not found")
            csv.writer(file, delimiter='\t').writerow([protein, sequence])

def create_unique_proteins_with_sequences_file(input_combined_file, fasta_file, output_unique_file):
    unique_proteins = get_unique_proteins(input_combined_file)
    protein_sequences = parse_fasta(fasta_file)
    write_protein_sequences(output_unique_file, unique_proteins, protein_sequences)

def get_unique_proteins(file_path):
    unique_proteins = set()
    with open(file_path, 'r', newline='') as file:
        for row in csv.reader(file, delimiter='\t'):
            if len(row) >= 2:
                unique_proteins.update(row[:2])
    return sorted(unique_proteins)

def parse_fasta(fasta_file):
    sequences = {}
    current_seq_id = None
    current_seq = []

    with open(fasta_file, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                if current_seq_id is not None:
                    sequences[current_seq_id] = ''.join(current_seq)
                current_seq_id = line[1:].split()[0]  # Extract sequence ID
                current_seq = []
            else:
                current_seq.append(line)
        
        # Add the last sequence
        if current_seq_id is not None:
            sequences[current_seq_id] = ''.join(current_seq)
    return sequences
    

def separate_files_based_on_sequence_length(intra, max_length, output_dir):
    # Read and classify protein sequences
    short_proteins, long_proteins = set(), set()
    interaction_file =output_dir+f"{intra}_interaction.tsv"
    dictionary_file = output_dir+f"{intra}_dictionary.tsv"
    with open(dictionary_file, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row[1]) <= max_length:
                short_proteins.add(row[0])
            else:
                long_proteins.add(row[0])

    # Helper function to filter interaction file based on protein set
    def filter_interaction_file(protein_set, output_file):
        with open(interaction_file, 'r') as f, open(output_file, 'w', newline='') as out:
            reader = csv.reader(f, delimiter='\t')
            writer = csv.writer(out, delimiter='\t')
            for row in reader:
                if row[0] in protein_set and row[1] in protein_set:
                    writer.writerow(row)

    # Filter and write interaction files
    filter_interaction_file(short_proteins, output_dir+f'{intra}_interaction_{max_length}_or_less.tsv')
    filter_interaction_file(long_proteins, output_dir+f'{intra}_interaction_{max_length}_greater.tsv')

    # Helper function to filter dictionary file based on protein set
    def filter_dictionary_file(protein_set, output_file):
        with open(dictionary_file, 'r') as f, open(output_file, 'w', newline='') as out:
            reader = csv.reader(f, delimiter='\t')
            writer = csv.writer(out, delimiter='\t')
            for row in reader:
                if row[0] in protein_set:
                    writer.writerow(row)

    # Filter and write dictionary files
    filter_dictionary_file(short_proteins, output_dir+f'{intra}_dictionary_{max_length}_or_less.tsv')
    filter_dictionary_file(long_proteins, output_dir+f'{intra}_dictionary_{max_length}_greater.tsv')
