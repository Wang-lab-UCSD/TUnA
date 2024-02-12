from model import (IntraEncoder, InterEncoder, ProteinInteractionNet, Tester)
from uncertaintyAwareDeepLearn import VanillaRFFLayer
from utils import (
    load_configuration,
    set_random_seed,
    get_computation_device,
    evaluate
)
import torch

def main():
    # --- Pre-Training Setup ---
    # Load configs. Use config file to change hyperparameters.
    config = load_configuration("config.yaml")
    
    # Set random seed for reproducibility
    set_random_seed(config['other']['random_seed'])
    
    # Determine the computation device (CPU or GPU)
    device = get_computation_device(config['other']['cuda_device'])
    
    # --- Model Initialization ---
    # Initialize the Encoder, Decoder, and overall model
    intra_encoder = IntraEncoder(config['model']['protein_embedding_dim'], config['model']['hid_dim'], config['model']['n_layers'], 
                      config['model']['n_heads'],config['model']['ff_dim'],config['model']['dropout'], config['model']['activation_function'], device)
    inter_encoder = InterEncoder(config['model']['protein_embedding_dim'], config['model']['hid_dim'], config['model']['n_layers'], 
                      config['model']['n_heads'], config['model']['ff_dim'], config['model']['dropout'], config['model']['activation_function'], device)
    gp_layer = VanillaRFFLayer(in_features=config['model']['hid_dim'], RFFs=config['model']['gp_layer']['rffs'], out_targets=config['model']['gp_layer']['out_targets'],
                               gp_cov_momentum=config['model']['gp_layer']['gp_cov_momentum'], gp_ridge_penalty=config['model']['gp_layer']['gp_ridge_penalty'], likelihood=config['model']['gp_layer']['likelihood_function'], random_seed=config['other']['random_seed'])
    model = ProteinInteractionNet(intra_encoder, inter_encoder, gp_layer, device)
    model.load_state_dict(torch.load(config['directories']['model_output'],map_location=device))
    model.eval()
    model.to(device)

    # Initialize the testing modules
    tester = Tester(model)
    
    # --- Evaluate trained model ---
    evaluate(config, tester)

# Execute the main function when the script is run
if __name__ == "__main__":
    main()
