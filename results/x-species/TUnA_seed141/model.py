import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torch.optim import Adam
import math
import numpy as np
from utils import *
from lookahead import Lookahead

class SelfAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        assert hid_dim % n_heads == 0, "hid_dim must be divisible by n_heads"

        # Linear transformations for query, key, and value
        self.w_q = spectral_norm(nn.Linear(hid_dim, hid_dim))
        self.w_k = spectral_norm(nn.Linear(hid_dim, hid_dim))
        self.w_v = spectral_norm(nn.Linear(hid_dim, hid_dim))

        # Final linear transformation
        self.fc = spectral_norm(nn.Linear(hid_dim, hid_dim))

        # Dropout for attention
        self.do = nn.Dropout(dropout)

        # Scaling factor for the dot product attention
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).to(device)

    def forward(self, query, key, value, mask=None):
        bsz = query.shape[0]

        # Compute query, key, value matrices [batch size, sent len, hid dim]
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        # Reshape for multi-head attention and permute to bring heads forward
        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)

        # Compute scaled dot-product attention
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # Apply mask if provided
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        # Compute attention weights [batch size, n heads, sent len_Q, sent len_K]
        attention = self.do(F.softmax(energy, dim=-1))
        
        # Apply attention to the value matrix
        x = torch.matmul(attention, V)

        # Reshape and concatenate heads
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))

        # Final linear transformation [batch size, sent len_Q, hid dim]
        x = self.fc(x)

        return x

class Feedforward(nn.Module):
    def __init__(self, hid_dim, ff_dim, dropout, activation_fn):
        super().__init__()

        self.hid_dim = hid_dim
        self.ff_dim = ff_dim

        self.fc_1 = spectral_norm(nn.Linear(hid_dim, ff_dim))  
        self.fc_2 = spectral_norm(nn.Linear(ff_dim, hid_dim))  

        self.do = nn.Dropout(dropout)
        self.activation = self._get_activation_fn(activation_fn)
    
    def _get_activation_fn(self, activation_fn):
        """Return the corresponding activation function."""
        if activation_fn == "relu":
            return nn.ReLU()
        elif activation_fn == "gelu":
            return nn.GELU()
        elif activation_fn == "elu":
            return nn.ELU()
        elif activation_fn == "swish":
            return nn.SiLU()
        elif activation_fn == "leaky_relu":
            return nn.LeakyReLU()
        elif activation_fn == "mish":
            return nn.Mish()
        # Add other activation functions if needed
        else:
            raise ValueError(f"Activation function {activation_fn} not supported.")
    
    def forward(self, x):
        # x = [batch size, sent len, hid dim]

        x = self.do(self.activation(self.fc_1(x)))
        # x = [batch size, ff dim, sent len]

        x = self.fc_2(x)
        # x = [batch size, hid dim, sent len]
        return x

class EncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, ff_dim, dropout, activation_fn, device):
        super().__init__()
        self.ln1 = nn.LayerNorm(hid_dim)
        self.ln2 = nn.LayerNorm(hid_dim)
        
        self.do1 = nn.Dropout(dropout)
        self.do2 = nn.Dropout(dropout)
        
        self.sa = SelfAttention(hid_dim, n_heads, dropout, device)
        self.ff = Feedforward(hid_dim, ff_dim, dropout, activation_fn)
        
    def forward(self, trg, mask=None):

        #trg_1 = trg
        #trg = self.sa(trg, trg, trg, trg_mask)
        #trg = self.ln1(trg_1 + self.do1(trg))
        #
        #trg = self.ln2(trg + self.do2(self.ff(trg)))

        trg = self.ln1(trg + self.do1(self.sa(trg, trg, trg, mask)))
        trg = self.ln2(trg + self.do2(self.ff(trg)))


        return trg

class IntraEncoder(nn.Module):
    def __init__(self, prot_dim, hid_dim, n_layers, n_heads, ff_dim, dropout, activation_fn, device):
        super().__init__()   
        self.ft = spectral_norm(nn.Linear(prot_dim, hid_dim))
        self.n_layers = n_layers
        self.layer = nn.ModuleList()
        for _ in range(n_layers):
            self.layer.append(EncoderLayer(hid_dim, n_heads, ff_dim, dropout, activation_fn, device))

        
    def forward(self, trg, trg_mask=None):
        # trg = [batch_size, max_seq_len, protA_dim]
        
        trg = self.ft(trg)
        # trg = [batch size, protA len, hid dim]

        for layer in self.layer:
            trg = layer(trg, trg_mask)
        return trg
  
class InterEncoder(nn.Module):
    """ protein feature extraction."""
    def __init__(self, prot_dim, hid_dim, n_layers, n_heads, ff_dim, dropout, activation_fn, device):
        super().__init__()
        self.output_dim = prot_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.device = device
        self.layer = nn.ModuleList()
        for _ in range(n_layers):
            self.layer.append(EncoderLayer(hid_dim, n_heads, ff_dim, dropout, activation_fn, device))



    def forward(self, enc_protA, enc_protB, combined_mask):
        # Concatenate the encoded representations and masks
        combined_trg_src = torch.cat([enc_protA, enc_protB], dim=1)

        for layer in self.layer:
            combined_trg_src = layer(combined_trg_src, combined_mask)

        combined_mask_2d = combined_mask[:,0,:,0]
        label = torch.sum(combined_trg_src*combined_mask_2d[:,:,None], dim=1)/combined_mask_2d.sum(dim=1, keepdims=True)
        
        return label 

class ProteinInteractionNet(nn.Module):
    def __init__(self, intra_encoder, inter_encoder, gp_layer, device):
        super().__init__()

        self.intra_encoder = intra_encoder
        self.inter_encoder = inter_encoder
        self.device = device
        self.gp_layer = gp_layer
        self.bce_loss = nn.BCELoss()

    
    def make_masks(self, prot_lens, protein_max_len):
        N = len(prot_lens)  # batch size
        mask = torch.zeros((N, protein_max_len, protein_max_len), device=self.device)

        for i, lens in enumerate(prot_lens):
            # Create a square mask for the non-padded sequences
            mask[i, :lens, :lens] = 1

        # Expand the mask to 4D: [batch, 1, max_len, max_len]
        mask = mask.unsqueeze(1)
        return mask

    def combine_masks(self, maskA, maskB):
        lenA, lenB = maskA.size(2), maskB.size(2)
        combined_mask = torch.zeros(maskA.size(0), 1, lenA + lenB, lenA + lenB, device=self.device)
        combined_mask[:, :, :lenA, :lenA] = maskA
        combined_mask[:, :, lenA:, lenA:] = maskB
        return combined_mask

    def mean_field_average(self, logits, variance):
        adjusted_score = logits / torch.sqrt(1. + (np.pi /8.)*variance)
        adjusted_score = torch.sigmoid(adjusted_score).squeeze()

        return adjusted_score

    def forward(self, protAs, protBs, protA_lens, protB_lens, batch_protA_max_length, batch_protB_max_length, last_epoch, train):

        protA_mask = self.make_masks(protA_lens, batch_protA_max_length)
        protB_mask = self.make_masks(protB_lens, batch_protB_max_length)

        enc_protA = self.intra_encoder(protAs, protA_mask)
        enc_protB = self.intra_encoder(protBs, protB_mask)
        
        combined_mask_AB = self.combine_masks(protA_mask, protB_mask)
        combined_mask_BA = self.combine_masks(protB_mask, protA_mask)

        AB_interaction = self.inter_encoder(enc_protA, enc_protB, combined_mask_AB)
        BA_interaction = self.inter_encoder(enc_protB, enc_protA, combined_mask_BA)
        
        #[batch, 64] 
        ppi_feature_vector, _ = torch.max(torch.stack([AB_interaction, BA_interaction], dim=-1), dim=-1)
        
        ### TRAINING ###
        # IF its not the last epoch, we don't need to update the precision
        if last_epoch==False and train==True:
            logit = self.gp_layer(ppi_feature_vector, update_precision=False)
            return logit
        # IF its the last epoch, we update precision
        elif last_epoch==True and train==True:
            logit =  self.gp_layer(ppi_feature_vector, update_precision=True)
            return logit

        ### TESTING ###
        # IF its not the last epoch, we don't need to get the variance
        elif last_epoch==False and train==False:
            logit = self.gp_layer(ppi_feature_vector, update_precision=False, get_var=False)
            return logit
        #This is the last test epoch. Generate variances.
        elif last_epoch==True and train==False:
            logit, var = self.gp_layer(ppi_feature_vector, update_precision=False, get_var=True)
            return logit, var

    
    def __call__(self, data, last_epoch, train):
        protAs, protBs, correct_interactions, protA_lens, protB_lens, batch_protA_max_length, batch_protB_max_length = data
        
        if train:
        # We don't use variances during training
            logit = self.forward(protAs, protBs, protA_lens, protB_lens, batch_protA_max_length, batch_protB_max_length, last_epoch, train=True)
            mean = torch.sigmoid(logit.squeeze())
            loss = self.bce_loss(mean, correct_interactions.float().squeeze())
            
            return loss

        #Test but not last epoch, we don't use variances still
        elif last_epoch==False and train==False:
            logit = self.forward(protAs, protBs, protA_lens, protB_lens, batch_protA_max_length, batch_protB_max_length, last_epoch, train=False)
            mean = torch.sigmoid(logit.squeeze())
            loss = self.bce_loss(mean, correct_interactions.float().squeeze())
            correct_labels = correct_interactions.cpu().data.numpy()
            
            return loss, correct_labels, mean
        
        #Test and last epoch
        elif last_epoch==True and train==False:
            logit, var = self.forward(protAs, protBs, protA_lens, protB_lens, batch_protA_max_length, batch_protB_max_length, last_epoch, train=False)
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

    def train(self, dataset, max_length, protein_dim, device, last_epoch):
        """ Train the model on the provided dataset. """
        self.model.train()
        self.optimizer.zero_grad()
        
        protAs, protBs, labels = zip(*dataset)
        data_pack = pack(protAs, protBs, labels, max_length, protein_dim, device)
        
        loss = self.model(data_pack, last_epoch, train=True)
        loss.backward()
        self.optimizer.step()
        
        return loss.item() * len(protAs)
    
class Tester(object):
    def __init__(self, model):
        self.model = model
    
    def test(self, dataset, max_length, protein_dim, last_epoch):
        """ Test the model on the provided dataset. """
        self.model.eval()

        with torch.no_grad():
            protAs, protBs, labels = zip(*dataset)
            data_pack = test_pack(protAs, protBs, labels, max_length, protein_dim, device=self.model.device)
            
            loss, correct_labels, adjusted_score = self.model(data_pack, last_epoch, train=False)
            T = correct_labels
            Y = np.round(adjusted_score.flatten().cpu().numpy())
            S = adjusted_score.flatten().cpu().numpy()
            
            return loss.item() * len(dataset), T, Y, S