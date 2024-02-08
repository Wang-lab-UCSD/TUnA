import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from utils import *
from lookahead import Lookahead
    
class Net(nn.Module):
    def __init__(self, protein_dim, hid_dim, dropout, device):
        super().__init__()
        self.device = device
        self.protein_dim = protein_dim
        self.hid_dim = hid_dim
        self.fc1 = nn.Linear(self.protein_dim*2,hid_dim)
        self.fc2 = nn.Linear(hid_dim,hid_dim)
        self.fc3 = nn.Linear(hid_dim,1)
        self.relu = nn.ReLU()
        self.bce_loss = nn.BCELoss()
        self.do = nn.Dropout(dropout)


    def forward(self, protAs, protBs):
        concatenated = torch.cat((protAs, protBs), dim=1)
        out = self.fc1(concatenated)
        out = self.relu(out)
        out = self.do(out)

        out = self.fc2(out)
        out = self.relu(out)
        out = self.do(out)

        out = self.fc3(out)

        return out
    
    def __call__(self, data):
        protAs, protBs, correct_interactions = data

        logit = self.forward(protAs, protBs)
        prediction = torch.sigmoid(logit.squeeze())
        loss = self.bce_loss(prediction, correct_interactions.float().squeeze())
        correct_labels = correct_interactions.cpu().data.numpy()
            
        return loss, correct_labels, prediction


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

    def train(self, dataset, protein_dim, device):
        """ Train the model on the provided dataset. """
        self.model.train()
        self.optimizer.zero_grad()
        
        protAs, protBs, labels = zip(*dataset)
        data_pack = pack(protAs, protBs, labels, protein_dim, device)
        
        loss, _, _ = self.model(data_pack)
        loss.backward()
        self.optimizer.step()
        
        return loss.item() * len(protAs)
    
class Tester(object):
    def __init__(self, model):
        self.model = model
    
    def test(self, dataset, protein_dim):
        """ Test the model on the provided dataset. """
        self.model.eval()

        with torch.no_grad():
            protAs, protBs, labels = zip(*dataset)
            data_pack = pack(protAs, protBs, labels, protein_dim, device=self.model.device)
            
            loss, correct_labels, score = self.model(data_pack)
            T = correct_labels
            Y = np.round(score.flatten().cpu().numpy())
            S = score.flatten().cpu().numpy()
            
            return loss.item() * len(dataset), T, Y, S
            