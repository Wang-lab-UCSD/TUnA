import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torch.optim import Adam
import math
import numpy as np
from utils import *
from lookahead import Lookahead
    
class ProteinInteractionNet(nn.Module):
    def __init__(self, protein_dim, hid_dim, dropout, gp_layer, device):
        super().__init__()
        self.protein_dim = protein_dim
        self.hid_dim = hid_dim
        self.gp_layer = gp_layer
        self.device = device
        self.fc1 = spectral_norm(nn.Linear(self.protein_dim*2, self.hid_dim))
        self.fc2 = spectral_norm(nn.Linear(self.hid_dim, self.hid_dim))
        self.relu = nn.ReLU()
        self.bce_loss = nn.BCELoss()
        self.do = nn.Dropout(dropout)
    
    def mean_field_average(self, logits, variance):
        adjusted_score = logits / torch.sqrt(1. + (np.pi /8.)*variance)
        adjusted_score = torch.sigmoid(adjusted_score).squeeze()

        return adjusted_score
    
    def forward(self, protAs, protBs, last_epoch, train):
        concatenated = torch.cat((protAs, protBs), dim=1)
        out = self.fc1(concatenated)
        out = self.relu(out)
        out = self.do(out)
        
        out = self.fc2(out)
        out = self.relu(out)
        out = self.do(out)
        
        ### TRAINING ###
        # IF its not the last epoch, we don't need to update the precision
        if last_epoch==False and train==True:
            logit = self.gp_layer(out, update_precision=False)
            return logit
        # IF its the last epoch, we update precision
        elif last_epoch==True and train==True:
            logit =  self.gp_layer(out, update_precision=True)
            return logit

        ### TESTING ###
        # IF its not the last epoch, we don't need to get the variance
        elif last_epoch==False and train==False:
            logit = self.gp_layer(out, update_precision=False, get_var=False)
            return logit
        #This is the last test epoch. Generate variances.
        elif last_epoch==True and train==False:
            logit, var = self.gp_layer(out, update_precision=False, get_var=True)
            return logit, var

    
    def __call__(self, data, last_epoch, train):
        protAs, protBs, correct_interactions = data
        
        if train:
        # We don't use variances during training
            logit = self.forward(protAs, protBs, last_epoch, train=True)
            mean = torch.sigmoid(logit.squeeze())
            loss = self.bce_loss(mean, correct_interactions.float().squeeze())
            
            return loss

        #Test but not last epoch, we don't use variances still
        elif last_epoch==False and train==False:
            logit = self.forward(protAs, protBs, last_epoch, train=False)
            mean = torch.sigmoid(logit.squeeze())
            loss = self.bce_loss(mean, correct_interactions.float().squeeze())
            correct_labels = correct_interactions.cpu().data.numpy()
            
            return loss, correct_labels, mean
        
        #Test and last epoch
        elif last_epoch==True and train==False:
            logit, var = self.forward(protAs, protBs, last_epoch, train=False)
            adjusted_score = self.mean_field_average(logit, var)
            loss = self.bce_loss(adjusted_score, correct_interactions.float().squeeze())
            correct_labels = correct_interactions.cpu().data.numpy()
            
            return loss, correct_labels, adjusted_score

class Trainer(object):
    def __init__(self, model, lr, weight_decay, batch):
        self.model = model
        self.batch = batch
        self._initialize_weights()
        self._setup_optimizer(lr, weight_decay)

    def _initialize_weights(self):
        """ Initialize model weights using Xavier Uniform Initialization for layers with dimension > 1. """
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _setup_optimizer(self, lr, weight_decay):
        """ Setup RAdam + Lookahead optimizer with separate weight decay for biases and weights. """
        weight_p, bias_p = self._separate_weights_and_biases()
        self.optimizer_inner = Adam(
            [{'params': weight_p, 'weight_decay': weight_decay}, {'params': bias_p, 'weight_decay': 0}], lr=lr)
        self.optimizer = Lookahead(self.optimizer_inner, alpha=0.8, k=5)

    def _separate_weights_and_biases(self):
        """ Separate model parameters into weights and biases. """
        weight_p, bias_p = [], []
        for name, p in self.model.named_parameters():
            if 'bias' in name:
                bias_p.append(p)
            else:
                weight_p.append(p)
        return weight_p, bias_p

    def train(self, dataset, protein_dim, device, last_epoch):
        """ Train the model on the provided dataset. """
        self.model.train()
        self.optimizer.zero_grad()
        
        protAs, protBs, labels = zip(*dataset)
        data_pack = pack(protAs, protBs, labels, protein_dim, device)
        
        loss = self.model(data_pack, last_epoch, train=True)
        loss.backward()
        self.optimizer.step()
        
        return loss.item() * len(protAs)
    
class Tester(object):
    def __init__(self, model):
        self.model = model
    
    def test(self, dataset, protein_dim, last_epoch):
        """ Test the model on the provided dataset. """
        self.model.eval()

        with torch.no_grad():
            protAs, protBs, labels = zip(*dataset)
            data_pack = pack(protAs, protBs, labels, protein_dim, device=self.model.device)
            
            loss, correct_labels, adjusted_score = self.model(data_pack, last_epoch, train=False)
            T = correct_labels
            Y = np.round(adjusted_score.flatten().cpu().numpy())
            S = adjusted_score.flatten().cpu().numpy()
            
            return loss.item() * len(dataset), T, Y, S