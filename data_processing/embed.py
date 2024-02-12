import torch
import os
import esm

def load_model_and_alphabet(device):
    """Load the ESM model and alphabet."""
    model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
    model.to(device)
    return model, alphabet

def get_protein_embeddings(model, alphabet, sequence, device):
    """Get protein embeddings from the ESM model."""
    batch_converter = alphabet.get_batch_converter()
    batch_labels, batch_strs, batch_tokens = batch_converter([("protein", sequence)])
    batch_tokens = batch_tokens.to(device)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[30], return_contacts=True)
        # Remove start and end tokens
        return results["representations"][30][0, 1:-1, :].to('cpu')

def process_data_points(model, alphabet, data_list, dir_input, device):
    """Process a list of data points and save protein embeddings."""
    protein_dictionary = {}
    for no, data in enumerate(data_list, 1):
        print(f"{no}/{len(data_list)}", flush=True)
        uniprot_id, sequence = data.strip().split("\t")
        protein_dictionary[uniprot_id] = get_protein_embeddings(model, alphabet, sequence, device)
    torch.save(protein_dictionary, os.path.join(dir_input, 'protein_dictionary.pt'))
