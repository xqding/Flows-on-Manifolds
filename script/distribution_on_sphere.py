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

xyz = vmf.sample(100000).numpy()
xyz = xyz.astype(np.float64)

angle = np.arcsin(xyz[:,-1])
dihedral = np.arctan2(xyz[:,1], xyz[:,0])

feature = np.stack([angle, dihedral], axis = -1)

with open("./output/train_data.pkl", 'wb') as file_handle:
    pickle.dump({'feature': feature, 'xyz': xyz}, file_handle)


fig = plt.figure(0, figsize = (6.4*2, 4.8))
fig.clf()
plt.subplot(1,2,1)
plt.hist(angle, density = True, range = [-np.pi/2, np.pi/2])
plt.subplot(1,2,2)
plt.hist(dihedral, density = True, range = [-np.pi, np.pi])
plt.savefig("./output/dist_angle_dihedral.pdf")

feature_dim = 2
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

# create model
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.set_default_tensor_type('torch.cuda.DoubleTensor')
else:
    device = torch.device('cpu')
    torch.set_default_tensor_type('torch.DoubleTensor')    

distribution = distributions.TweakedUniform(torch.tensor([-np.pi]*feature_dim), torch.tensor([np.pi]*feature_dim))
flow = flows.Flow(transform, distribution).to(device)

feature = torch.from_numpy(feature).to(device)

optimizer = optim.Adam(flow.parameters(), lr=1e-3)
scheduler = MultiStepLR(optimizer, [1000, 2000], gamma=0.5, last_epoch=-1)

batch_size = 10240
num_steps = 3000

start_time = time.time()
loss_record = []
for idx_step in range(num_steps + 1):
    flow.train()
    optimizer.zero_grad()

    indices = np.random.choice(feature.shape[0], size = batch_size, replace = False)

    batch_feature = feature[indices]
    batch_context = None
    
    log_density = flow.log_prob(batch_feature, batch_context)
    loss = -torch.mean(log_density)
    
    loss.backward()
    optimizer.step()
    scheduler.step()    
    loss_record.append(loss.item())

    if (idx_step + 1) % 10 == 0:
        lr = optimizer.param_groups[0]['lr']
        print("idx_steps: {:}, lr: {:.5f}, loss: {:.5f}".format(idx_step, lr, loss.item()), flush = True)

    if (idx_step + 1) % 100 == 0:
        print("time used for 100 steps: {:.3f}".format(time.time() - start_time))
        start_time = time.time()

print("model saved at step: {} with loss: {:.5f}".format(idx_step, loss.item()), flush = True)
torch.save({'state_dict': flow.state_dict(),
            'loss_training': loss.item()},
           "./output/flow_step_{}.pt".format(idx_step))
    

