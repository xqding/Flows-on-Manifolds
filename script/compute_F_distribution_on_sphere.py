__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2020/07/09 23:30:53"

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import torch
torch.set_default_dtype(torch.double)
import tensorflow as tf
import tensorflow_probability as tfp
from torch.optim.lr_scheduler import MultiStepLR
import math
import sys
sys.path.append("/home/xqding/course/projectsOnGitHub/nsf")
import argparse
from nde import distributions, flows, transforms
import utils
import nn as nn_
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
import pickle
import time
import argparse
from sys import exit
from FastMBAR import *


parser = argparse.ArgumentParser()
parser.add_argument('--hidden_features', type=int, default=4,
                    help='Number of hidden features to use in coupling/autoregressive nets.')
parser.add_argument('--num_flow_steps', type=int, default=4,
                    help='Number of blocks to use in flow.')
parser.add_argument('--num_transform_blocks', type=int, default=2,
                    help='Number of blocks to use in coupling/autoregressive nets.')
parser.add_argument('--dropout_probability', type=float, default=0.00,
                    help='Dropout probability for coupling/autoregressive nets.')
parser.add_argument('--use_batch_norm', type=int, default=0,
                    choices=[0, 1],
                    help='Whether to use batch norm in coupling/autoregressive nets.')
parser.add_argument('--num_bins', type=int, default=3,
                    help='Number of bins to use for piecewise transforms.')
parser.add_argument('--apply_unconditional_transform', type=int, default=0,
                    choices=[0, 1],
                    help='Whether to unconditionally transform \'identity\' '
                         'features in coupling layer.')
args = parser.parse_args()

## mean direction of vmf distribution
mu_1, mu_2 = 0.2, 0.3
mu_3 = math.sqrt(1 - mu_1**2 - mu_2**2)
mu = [mu_1, mu_2, mu_3]

## concentration 
conc = 2.0

## define the vmf distribution
vmf = tfp.distributions.VonMisesFisher(mean_direction = mu,
                                       concentration = conc)

with open("./output/train_data.pkl", 'rb') as file_handle:
    data = pickle.load(file_handle)    
    xyz = data['xyz']
    feature = data['feature']
    
feature_dim = feature.shape[-1]
context_dim = None

def create_linear_transform():
    linear_transform = transforms.CompositeTransform([
        transforms.RandomPermutation(features=feature_dim)
    ])
    return linear_transform

def create_base_transform(i):
    base_transform = transforms.PiecewiseCircularRationalQuadraticCouplingTransform_edge(
        mask=utils.create_alternating_binary_mask(feature_dim, even=(i % 2 == 0)),
        transform_net_create_fn=lambda in_features, out_features: nn_.ResidualNet(
            in_features=in_features,
            out_features=out_features,
            hidden_features=args.hidden_features,
            context_features=context_dim,
            num_blocks=args.num_transform_blocks,
            activation=F.relu,
            dropout_probability=args.dropout_probability,
            use_batch_norm=args.use_batch_norm
        ),
        num_bins=args.num_bins,
        tails=None,
        apply_unconditional_transform=args.apply_unconditional_transform
    )
    return base_transform

transform = transforms.CompositeTransform(
    [ transforms.CompositeTransform([create_linear_transform(),
                                     create_base_transform(i)])
      for i in range(args.num_flow_steps)] +
    [create_linear_transform()])


device = torch.device('cpu')
torch.set_default_tensor_type('torch.DoubleTensor')    

base_dist = distributions.TweakedUniform(torch.tensor([-np.pi]*feature_dim), torch.tensor([np.pi]*feature_dim))
flow = flows.Flow(transform, base_dist).to(device)

num_steps = 3000
data = torch.load("./output/flow_step_{}.pt".format(num_steps),
                  map_location=torch.device('cpu'))
flow.load_state_dict(data['state_dict'])
flow.eval()

## compute pq_logp and pq_logq
pq_logp = vmf.log_prob(xyz).numpy() + np.log(np.abs(np.cos(feature[:,0])))
with torch.no_grad():
    pq_logq = flow.log_prob(torch.from_numpy(feature), None).numpy()

## compute qp_logq and qp_logp
N = feature.shape[0]
z = base_dist.sample(N, None)

with torch.no_grad():
    feature_q, logJ = flow._transform.inverse(z, None)
    qp_logq = flow.log_prob(feature_q, None).numpy()
    
xyz_q = np.zeros((N, 3))
xyz_q[:,2] = np.sin(feature_q[:,0])
xyz_q[:,0] = np.cos(feature_q[:,0])*np.cos(feature_q[:,1])
xyz_q[:,1] = np.cos(feature_q[:,0])*np.sin(feature_q[:,1])

qp_logp = vmf.log_prob(xyz_q).numpy() + np.log(np.abs(np.cos(feature_q.numpy()[:,0])))

logq = np.concatenate([qp_logq, pq_logq])
logp = np.concatenate([qp_logp, pq_logp])
energy_matrix = -np.stack([logq, logp])

num_conf = np.array([len(qp_logq), len(pq_logq)])

fastmbar = FastMBAR(energy_matrix, num_conf, verbose = True)

