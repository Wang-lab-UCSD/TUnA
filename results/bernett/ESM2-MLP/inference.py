import torch
from model import (Net, Tester)
from utils import (
    load_configuration,
    initialize_logging,
    set_random_seed,
    get_computation_device,
    load_and_predict
)


def main():
    # --- Pre-Training Setup ---
    # Load configs. Use config file to change hyperparameters.
    config = load_configuration("config.yaml")
    
    # Set up logging to save output to a text file
    #initialize_logging("output/inference_results.txt")
    
    # Set random seed for reproducibility
    set_random_seed(config['other']['random_seed'])
    
    # Determine the computation device (CPU or GPU)
    device = get_computation_device(config['other']['cuda_device'])
    
    # --- Model Initialization ---
    # Initialize the Encoder, Decoder, and overall model
    model = Net(config['model']['protein_embedding_dim'], config['model']['hid_dim'],config['model']['dropout'], device)
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
