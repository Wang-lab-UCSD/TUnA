import torch
import random
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, auc, precision_recall_curve, precision_score
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import calibration as cal
import logging
import yaml
import os
import pandas as pd
import numpy as np
import csv
import timeit

# ------------------- Initialization and Configuration -------------------
# Set up logging to a specific file
def initialize_logging(log_file):
    logging.basicConfig(filename=log_file, filemode='w', level=logging.INFO, format='%(message)s')
    logging.info("Epoch Time              Train Loss          Test Loss           AUC                 PRC                 Accuracy            Sensitivity         Specificity         Precision           F1                  MCC                 Max AUC")

# Set random seed for reproducibility
def set_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

# Load configuration settings from a YAML file
def load_configuration(config_file):
    with open(config_file, 'r') as config_file:
        return yaml.safe_load(config_file)

# Determine which device to use for computation (CPU or CUDA)
def get_computation_device(cuda_device=None):
    if cuda_device is not None:
        return torch.device(cuda_device)
    else:
        logging.info('Warning: Running on CPU')
        return torch.device('cpu')

# ------------------- Model Initialization -------------------
# Initialize the learning rate scheduler
def initialize_scheduler(trainer, optimizer_config):
    return torch.optim.lr_scheduler.StepLR(trainer.optimizer, optimizer_config['step_size'], optimizer_config['gamma'])


# ------------------- Model Training and Testing -------------------
# Train the model for one epoch
def train_epoch(dictionary, action_file, subset, trainer, config, device, last_epoch):
    total_loss = 0
    total_samples = 0
    batch_size = config['training']['batch_size']
    max_seq_length = config['model']['max_sequence_length']
    protein_dim = config['model']['protein_embedding_dim']    
    
    dataset = ProteinInteractionDataset(dictionary, action_file, subset)
    total_samples += len(dataset)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=list_collate)
    
    for proteinA, proteinB, labels in train_loader:
        dataset_batch = list(zip(proteinA, proteinB, labels))
        batch_loss = trainer.train(dataset_batch, max_seq_length, protein_dim, device, last_epoch)
        
        total_loss += batch_loss

    return total_loss, total_samples

# Test the model for one epoch
def test_epoch(dictionary, action_file, subset, tester, config, last_epoch):
    
    T, Y, S = [], [], []
    total_loss = 0
    total_samples = 0
    max_seq_length = config['model']['max_sequence_length']
    protein_dim = config['model']['protein_embedding_dim'] 
    dataset = ProteinInteractionDataset(dictionary, action_file, subset)
    total_samples += len(dataset)
    dev_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=list_collate)
    
    if last_epoch: 
        for proteinA, proteinB, labels in dev_loader:
            dataset_batch = list(zip(proteinA, proteinB, labels))
            batch_loss, t, y, s = tester.test(dataset_batch, max_seq_length, protein_dim, last_epoch)
            T.extend(t)
            Y.extend(y)
            S.extend(s)
            total_loss += batch_loss
            
        return T, Y, S, total_loss, total_samples

    else:
        for proteinA, proteinB, labels in dev_loader:
            dataset_batch = list(zip(proteinA, proteinB, labels))
            batch_loss, t, y, s = tester.test(dataset_batch, max_seq_length, protein_dim, last_epoch)
            T.extend(t)
            Y.extend(y)
            S.extend(s)
            total_loss += batch_loss
            
        return T, Y, S, total_loss, total_samples

# Train and validate the model across multiple epochs
def train_and_validate_model(config, trainer, tester, scheduler, model, device):
    max_AUC_dev = 0
    train_dictionary = load_dictionary(config['directories']['train_dictionary'])
    test_dictionary = load_dictionary(config['directories']['validation_dictionary'])

    train_interactions = config['directories']['train_interactions']
    test_interactions = config['directories']['validation_interactions']
    start = timeit.default_timer()
    subset = config['training']['subset']
    if subset < 0:
        subset = None

    for epoch in range(1, config['training']['iteration'] + 1):
        if epoch != (config['training']['iteration']):
            total_loss_train, total_train_size = train_epoch(train_dictionary, train_interactions, subset, trainer, config, device, last_epoch=False)
            T, Y, S, total_loss_test, total_test_size = test_epoch(test_dictionary, test_interactions, subset, tester, config, last_epoch=False)
            
            end = timeit.default_timer()
            time = end - start

            AUC_dev, PRC_dev, accuracy, sensitivity, specificity, precision, f1, mcc = calculate_metrics(T,Y,S)
            
            if AUC_dev > max_AUC_dev:
                save_model(model, "output/model")
                max_AUC_dev = AUC_dev
            
            log_and_save_metrics(epoch, time, total_loss_train, total_train_size, total_loss_test, total_test_size, AUC_dev, PRC_dev, accuracy, sensitivity, specificity, precision, f1, mcc, max_AUC_dev)
            scheduler.step()
            plot(config['directories']['metrics_output'])

        
        if epoch == (config['training']['iteration']):
            total_loss_train, total_train_size = train_epoch(train_dictionary, train_interactions, subset, trainer, config, device, last_epoch=True)
            
            T, Y, S, total_loss_test, total_test_size= test_epoch(test_dictionary, test_interactions, subset, tester, config, last_epoch=True)
            AUC_dev, PRC_dev, accuracy, sensitivity, specificity, precision, f1, mcc = calculate_metrics(T,Y,S)
            
            end = timeit.default_timer()
            time = end - start
            log_and_save_metrics(epoch, time, total_loss_train, total_train_size, total_loss_test, total_test_size, AUC_dev, PRC_dev, accuracy, sensitivity, specificity, precision, f1, mcc, max_AUC_dev)
            plot(config['directories']['metrics_output'])
            save_model(model, "output/model")

def evaluate(config, tester):
    test_dictionary = load_dictionary(config['directories']['test_dictionary'])
    test_interactions = config['directories']['test_interactions']

    subset = config['training']['subset']
    if subset < 0:
        subset = None

    T, Y, S, total_loss_test, total_test_size = test_epoch(test_dictionary, test_interactions, subset, tester, config, last_epoch=True)
    AUC_dev, PRC_dev, accuracy, sensitivity, specificity, precision, f1, mcc = calculate_metrics(T, Y, S)
    print(total_loss_test / total_test_size, AUC_dev, PRC_dev, accuracy, sensitivity, specificity, precision, f1, mcc)

    # Calculate Expected Calibration Error
    ece = cal.get_ece(S, T)
    print("Expected Calibration Error (ECE):", ece)
    
    # Calculate uncertainty
    uncertainty = (1 - np.array(S)) * (np.array(S)) / 0.25

    # Add S and uncertainty columns to test_interactions DataFrame
    test_interactions = pd.read_csv(test_interactions, sep='\t', header=None)
    column_names = ['Protein A', 'Protein B', 'T']
    test_interactions.columns = column_names[:len(test_interactions.columns)]
    test_interactions['S'] = S
    test_interactions['uncertainty'] = uncertainty

    # Saving to TSV
    test_interactions.to_csv('evaluation_results.tsv', sep='\t', index=False)

    for cutoff in [0.2, 0.4, 0.6, 0.8]:
        filtered_indices = uncertainty < cutoff
        T_filtered = np.array(T)[filtered_indices]
        Y_filtered = np.array(Y)[filtered_indices]
        true_positives = sum((T_filtered == 1) & (Y_filtered == 1))
        precision_filtered = precision_score(T_filtered, Y_filtered, zero_division=0)
        print(f"Uncertainty Cutoff {cutoff}: Precision - {precision_filtered}, True Positives - {true_positives}")

# ------------------- Data Loading -------------------

class ProteinInteractionDataset(Dataset):
    def __init__(self, protein_dict, protein_interactions_file, limit=None):
        self.protein_dict = protein_dict
        self.protein_interactions = []
        with open(protein_interactions_file, 'r') as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                protein_a, protein_b, label = line.strip().split("\t")
                self.protein_interactions.append((protein_a, protein_b, int(label)))

    def __len__(self):
        return len(self.protein_interactions)

    def __getitem__(self, index):
        protein_a, protein_b, label = self.protein_interactions[index]
        return self.protein_dict[protein_a], self.protein_dict[protein_b], label

def load_dictionary(protein_dictionary_file):
    return torch.load(protein_dictionary_file)

# Custom collate function for DataLoader
def list_collate(batch):
    proteinA_batch = [item[0] for item in batch]
    proteinB_batch = [item[1] for item in batch]
    y_batch = [item[2] for item in batch]
    return proteinA_batch, proteinB_batch, y_batch


# ------------------- Metrics and Logging -------------------
def calculate_metrics(T, Y, S):
    AUC_dev = roc_auc_score(T, S)
    tpr, fpr, _ = precision_recall_curve(T, S)
    PRC_dev = auc(fpr, tpr)
    accuracy = accuracy_score(T, Y)
    tn, fp, fn, tp = confusion_matrix(T, Y).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp)
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity)
    mcc = (tp * tn - fp * fn) / ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    return AUC_dev, PRC_dev, accuracy, sensitivity, specificity, precision, f1, mcc

# Log and save metrics
def log_and_save_metrics(epoch, time, total_loss_train, total_train_size, total_loss_test, total_test_size, AUC_dev, PRC_dev, accuracy, sensitivity, specificity, precision, f1, mcc, max_AUC_dev):
    metrics = [epoch, time, total_loss_train/total_train_size, total_loss_test/total_test_size, AUC_dev, PRC_dev,accuracy, sensitivity, specificity, precision, f1, mcc, max_AUC_dev]
    logging.info('\t'.join(map(str, metrics)))

# Save model state to file
def save_model(model, filename):
    torch.save(model.state_dict(), filename)

# Plot metrics across epochs
def plot(directory, train=True):
    train_loss = []
    test_loss = []
    AUC_dev = []
    PRC_dev = []
    acc = []
    sens = []
    spec = []
    prec = []
    f1 = []
    mcc = []
    max_auc = []
    
    # Open output and extract columns
    with open(directory, newline='') as file:
        reader = csv.reader(file, delimiter='\t')
        next(reader)
        for row in reader:
            train_loss.append(float(row[2]))
            test_loss.append(float(row[3]))
            AUC_dev.append(float(row[4]))
            PRC_dev.append(float(row[5]))
            acc.append(float(row[6]))
            sens.append(float(row[7]))
            spec.append(float(row[8]))
            prec.append(float(row[9]))
            f1.append(float(row[10]))
            mcc.append(float(row[11]))
            max_auc.append(float(row[12]))

    # Get the current directory name
    current_dir = os.path.basename(os.getcwd())
    plt.close()
    plt.figure(figsize=(8, 10))
    plt.subplot(2, 1, 1)
    # Plot loss
    plt.plot(train_loss, label='Training Loss')
    if train:
        plt.plot(test_loss, label='Validation Loss')
    else: 
        plt.plot(test_loss, label='Test Loss')
    plt.legend()
    plt.title(current_dir[:2])  # Print only the first two letters
    plt.ylabel('Loss')
    plt.grid()
    plt.tick_params(labelbottom=False)

    # Plotting metrics
    plt.subplot(2, 1, 2)
    plt.plot(AUC_dev, label='AUC')
    plt.plot(PRC_dev, label='PRC')
    plt.plot(acc, label='Accuracy')
    plt.plot(sens, label='Sensitivity')
    plt.plot(spec, label='Specificity')
    plt.plot(prec, label='Precision')    
    plt.plot(f1, label='F1')
    plt.plot(mcc, label='MCC')
    plt.plot(max_auc, label='Max Auc')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Metrics')
    plt.grid()

    plt.tight_layout()
    plt.savefig(current_dir[:2] + '.png', dpi=300)
    plt.close()

# ------------------- Data Manipulation -------------------
# Prepare data for training/testing (packing, pooling, etc.)
def pack(protAs, protBs, labels, max_length, protein_dim, device):

    N = len(protAs)
    protA_lens = [protA.shape[0] for protA in protAs]
    protB_lens = [protB.shape[0] for protB in protBs]

    #batch_max_length = min(max(max(protA_lens), max(protB_lens)),max_length)
    batch_max_length = max_length


    protAs_new = torch.zeros((N, batch_max_length, protein_dim), device=device)
    for i, protA in enumerate(protAs):
        a_len = protA.shape[0]
        if a_len <= batch_max_length:
            protAs_new[i, :a_len, :] = protA
        else:
            start_pos = random.randint(0,a_len-batch_max_length)
            protAs_new[i, :batch_max_length, :]  = protA[start_pos:start_pos+batch_max_length]
    
    protBs_new = torch.zeros((N, batch_max_length, protein_dim), device=device)
    for i, protB in enumerate(protBs):
        b_len = protB.shape[0]
        if b_len <= batch_max_length:
            protBs_new[i, :b_len, :] = protB
        else:
            start_pos = random.randint(0,b_len-batch_max_length)
            protBs_new[i, :batch_max_length, :]  = protB[start_pos:start_pos+batch_max_length]
    # labels_new: torch.tensor [N,]
    
    labels_new = torch.zeros(N, dtype=torch.long, device=device)
    for i, label in enumerate(labels):
        # Convert the label (assuming it's a NumPy array) to a PyTorch tensor
        labels_new[i] = label

    return (protAs_new, protBs_new, labels_new, protA_lens, protB_lens, batch_max_length, batch_max_length)

def test_pack(protAs, protBs, labels, max_length, protein_dim, device):

    N = len(protAs)
    protA_lens = [protA.shape[0] for protA in protAs]
    protB_lens = [protB.shape[0] for protB in protBs]

    batch_protA_max_length = max(protA_lens)
    batch_protB_max_length = max(protB_lens)

    protAs_new = torch.zeros((N, batch_protA_max_length, protein_dim), device=device)
    for i, protA in enumerate(protAs):
        a_len = protA.shape[0]
        protAs_new[i, :a_len, :] = protA

    protBs_new = torch.zeros((N, batch_protB_max_length, protein_dim), device=device)
    for i, protB in enumerate(protBs):
        b_len = protB.shape[0]
        protBs_new[i, :b_len, :] = protB

    # labels_new: torch.tensor [N,]
    
    labels_new = torch.zeros(N, dtype=torch.long, device=device)
    for i, label in enumerate(labels):
        # Convert the label (assuming it's a NumPy array) to a PyTorch tensor
        labels_new[i] = label

    return (protAs_new, protBs_new, labels_new, protA_lens, protB_lens, batch_protA_max_length, batch_protB_max_length)
