from model import (ProteinInteractionNet, Trainer, Tester)
from uncertaintyAwareDeepLearn import VanillaRFFLayer
from utils import (
    load_configuration,
    initialize_logging,
    set_random_seed,
    get_computation_device,
    initialize_scheduler,
    train_and_validate_model
)

def main():
    # --- Pre-Training Setup ---
    # Load configs. Use config file to change hyperparameters.
    config = load_configuration("config.yaml")
    
    # Set up logging to save output to a text file
    initialize_logging("output/results.txt")
    
    # Set random seed for reproducibility
    set_random_seed(config['other']['random_seed'])
    
    # Determine the computation device (CPU or GPU)
    device = get_computation_device(config['other']['cuda_device'])
    
    # --- Model Initialization ---
    gp_layer = VanillaRFFLayer(in_features=config['model']['hid_dim'], RFFs=config['model']['gp_layer']['rffs'], out_targets=config['model']['gp_layer']['out_targets'],
                               gp_cov_momentum=config['model']['gp_layer']['gp_cov_momentum'], gp_ridge_penalty=config['model']['gp_layer']['gp_ridge_penalty'], likelihood=config['model']['gp_layer']['likelihood_function'], random_seed=config['other']['random_seed'])
    model = ProteinInteractionNet(config['model']['protein_embedding_dim'], config['model']['hid_dim'],config['model']['dropout'], gp_layer, device)
    model.to(device)
    
    # Initialize the training and testing modules
    trainer = Trainer(model, config['training']['learning_rate'], config['training']['weight_decay'], config['training']['batch_size'])
    tester = Tester(model)
    scheduler = initialize_scheduler(trainer, config['optimizer'])
    
    # --- Training and Validation ---
    # Perform training and validation
    train_and_validate_model(config, trainer, tester, scheduler, model, device)

# Execute the main function when the script is run
if __name__ == "__main__":
    main()
