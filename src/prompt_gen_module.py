#add the absolute path of openprompt library
import sys, os
import math

sys.path.append('../OpenPrompt/')

from torch.utils.data.sampler import RandomSampler
from transformers.configuration_utils import PretrainedConfig
from transformers.generation_utils import GenerationMixin
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import *
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils.dummy_pt_objects import PreTrainedModel
import numpy as np
from torch.utils.data import DataLoader
from transformers import  AdamW, get_linear_schedule_with_warmup
import torch.nn.functional as F
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer
from transformers import T5Tokenizer, T5EncoderModel
from torch.distributions.multinomial import Multinomial


class PromptGenerator(nn.Module):
    '''
        CQ-module
    '''
    def __init__(self, 
                plm: PreTrainedModel,
                num_codes: int,
                num_cq_tokens: int,
                num_samples: int,
                temp=float,
                padding_idx = None,
                commitment_cost = float,
                ema_decay=0.99,
                epsilon=1e-2,
                centroid_warm_up=False,
                calc_k_means=False
                ):

        super(PromptGenerator, self).__init__()

        self.raw_embedding = plm.get_input_embeddings()
        self.embedding_dim = self.raw_embedding.weight.size(1)
        self.num_cq_tokens = num_cq_tokens
        self.num_codes = num_codes
        self.cls_embeds = None

        self.d_model = self.embedding_dim
        self.codebook_size = num_codes
        self.padding_idx = padding_idx

        #average perplexity
        self.smooth_avg_perplexity = []

        #transformer encoder
        self.fc_in = nn.Linear(768, 32, bias=True)
        self.fc_out = nn.Linear(32, 768, bias=True)
        self.encoder_layer = TransformerEncoderLayer(d_model=32, 
                                                    nhead=4,
                                                    dim_feedforward=2*32) #2048
        self.sentence_encoder = TransformerEncoder(self.encoder_layer, num_layers=2)

        self.codebook = nn.Embedding.from_pretrained(self.raw_embedding.weight[self.num_cq_tokens:self.num_cq_tokens+self.codebook_size].clone().detach(), freeze=False)
        self.codebook.weight.data = 100*torch.nn.functional.normalize(self.codebook.weight.data, dim=-1)

        if padding_idx is not None:
            self.codebook.weight.data[padding_idx] = 0

        self.commitment_cost = commitment_cost
        self.temp = temp
        self.num_samples = num_samples

        self.register_buffer('_ema_cluster_size', torch.ones(self.codebook_size)/self.num_codes)

        self._decay = ema_decay
        self._epsilon = epsilon
        self.discard_ema_cluster_sizes = False

        self.loss_aux = torch.tensor(0.0)
        self.noise_contrastive_loss = False
        self.dedicated_codebook = False
    
    def forward(self, xcq, commitment_cost=None, attn_mask=None, temp=None):
        device = xcq.device

        #encode the input
        batch_size = xcq.size(0)

        #input text
        xc_down = self.fc_in(xcq)
        xc_down = xc_down.transpose(0,1)
        
        attn_mask_se = torch.cat([torch.ones(batch_size, self.num_cq_tokens).to(device=xcq.device), attn_mask], dim=-1).clone().detach()
        attn_mask_se = attn_mask_se.bool()
        
        xc_up = self.sentence_encoder( src=xc_down , src_key_padding_mask = ~attn_mask_se ).transpose(0,1)
        xc_out = self.fc_out(xc_up[:,:self.num_cq_tokens,:])

        #quantizer
        if commitment_cost is None: #a hyperparameter need to tune
            commitment_cost = self.commitment_cost

        if temp is None:
            temp = self.temp

        xc_out_shape = xc_out.size()
        xc_out_dims = xc_out.dim()

        # Flatten input
        flat_xc_out = xc_out.reshape(-1, self.d_model)

        # calculate distances
        if self.dedicated_codebook:
            for d in range(self.num_cq_tokens):
                distances = distances.view(xc_out_shape[0], xc_out_dims[1], self.num_codes)
                distances[:, d, :] += 1e5
                distances[:, d, 10*d : 10*(d+1)] -= 1e5
            distances = distances.reshape(-1, self.num_codes)
        else:
            distances = (torch.sum(flat_xc_out**2, dim=1, keepdim=True) 
                        + torch.sum(self.codebook.weight**2, dim=1)
                        - 2 * torch.matmul(flat_xc_out, self.codebook.weight.t()))

        # Define multinomial distribution and sample from it
        multi = Multinomial(total_count=self.num_samples, logits=(-distances-1e-5)/temp)
        samples = multi.sample().to(device)

        # Soft-quantize and unflatten
        xc_quantized = torch.matmul(samples, self.codebook.weight).view(xc_out_shape) / self.num_samples

        # Loss
        e_latent_loss = torch.mean((xc_quantized.detach() - xc_out)**2)
        loss = commitment_cost * e_latent_loss
        
        # Use EMA to update the embedding vectors
        if self.training:
            if self.discard_ema_cluster_sizes:
                self._ema_cluster_size = torch.sum(samples, 0) / self.num_samples
                self.discard_ema_cluster_sizes = False
            else:
                self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                         (1 - self._decay) * \
                                         (torch.sum(samples, 0) / self.num_samples)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self.codebook_size * self._epsilon) * n)

            dw = torch.matmul(samples.t(), flat_xc_out) / self.num_samples
            normalized_ema_w = self.codebook.weight * self._decay + (1 - self._decay) * (dw/self._ema_cluster_size.unsqueeze(1)) #option-1
            
            if self.padding_idx is not None:
                normalized_ema_w[self.padding_idx] = 0
            self.codebook.weight = nn.Parameter(normalized_ema_w)

        xc_quantized = xc_out + (xc_quantized - xc_out).detach()
        avg_probs = torch.mean(samples, dim=0) / self.num_samples
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))# e^(10*log_e(10)))

        samples = samples.reshape(list(xc_out_shape[:xc_out_dims - 1]) + [self.codebook_size])

        #check the perplexity over the entire epoch
        self.smooth_avg_perplexity.append(perplexity.detach())
        print("-->e_latent_loss", e_latent_loss.item())
        print("-->perplexity:", (sum(self.smooth_avg_perplexity[-100:]).detach()/len(self.smooth_avg_perplexity[-100:])).item() )
        self.loss_aux = loss

        if self.noise_contrastive_loss:
            xc_out2 = self.sentence_encoder( src=xc_down, src_key_padding_mask = ~attn_mask_se ).transpose(0,1)
            xc_out2 = self.fc_out(xc_out2[:,:self.num_cq_tokens,:])
            dist = torch.cdist(xc_out, xc_out2, p=2)
            loss_contrastive = torch.softmax(-dist, dim=-1)
            loss_contrastive = -torch.diagonal(loss_contrastive, dim1=-2, dim2=-1)
            loss_contrastive = torch.mean(loss_contrastive)
            print("-->se loss contrastive:", loss_contrastive.item())
            self.loss_aux += loss_contrastive

        return xc_quantized, xcq

