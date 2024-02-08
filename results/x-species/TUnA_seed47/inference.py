from model import (IntraEncoder, InterEncoder, Net, Trainer, Tester)
from uncertaintyAwareDeepLearn import myVanillaRFFLayer
from utils import (
    load_configuration,
    initialize_logging,
    set_random_seed,
    get_computation_device,
    load_and_predict
)
import torch

def main():
    # --- Pre-Training Setup ---
    # Load configs. Use config file to change hyperparameters.
    config = load_configuration("config.yaml")
    
    # Set up logging to save output to a text file
    #initialize_logging("output/results.txt")
    
    # Set random seed for reproducibility
    set_random_seed(config['other']['random_seed'])
    
    # Determine the computation device (CPU or GPU)
    device = get_computation_device(config['other']['cuda_device'])
    
    # --- Model Initialization ---
    # Initialize the Encoder, Decoder, and overall model
    intra_encoder = IntraEncoder(config['model']['protein_embedding_dim'], config['model']['hid_dim'], config['model']['n_layers'], 
                      config['model']['n_heads'],config['model']['conv1d_dim'],config['model']['dropout'], config['model']['activation_function'], device)
    inter_encoder = InterEncoder(config['model']['protein_embedding_dim'], config['model']['hid_dim'], config['model']['n_layers'], 
                      config['model']['n_heads'], config['model']['conv1d_dim'], config['model']['dropout'], config['model']['activation_function'], device)
    gp_layer = myVanillaRFFLayer(config['model']['hid_dim'], config['model']['gp_layer']['rffs'], config['model']['gp_layer']['out_targets'], config['model']['gp_layer']['kernel_amplitude'],
                               config['model']['gp_layer']['gp_cov_momentum'], config['model']['gp_layer']['gp_ridge_penalty'], config['model']['gp_layer']['likelihood_function'], config['other']['random_seed'])
    model = Net(intra_encoder, inter_encoder, gp_layer, device)
    model.load_state_dict(torch.load(config['directories']['model_output']))
    model.eval()
    model.to(device)

    # Initialize the testing modules
    tester = Tester(model)

    # --- Training and Validation ---
    # Perform training and validation
    load_and_predict(config, tester, model, device)

# Execute the main function when the script is run
if __name__ == "__main__":
    main()
