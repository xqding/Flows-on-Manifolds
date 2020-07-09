__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2020/05/19 17:35:46"

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.distributions as distributions
import sys
sys.path.append("./script/")
from mobius_transform import *
import torch.optim as optim
import scipy.stats as stats

w = torch.tensor([-2, -5.])
mt = mobius_transform(w)

optimizer = optim.Adam(mt.parameters(), lr = 0.001)
kappa = 3
loc = 1.0

for idx_step in range(10000):
    radius = stats.vonmises.rvs(kappa = kappa, loc = loc, size = 1024)    
    radius[radius < 0] = radius[radius < 0] + 2*math.pi
    radius[radius > 2*math.pi] = radius[radius > 2*math.pi] - 2*math.pi
    
    radius = torch.from_numpy(radius)
    radius.requires_grad = True
    
    base_radius, jacobian = mt.reverse(radius)
    loss = -torch.mean(torch.log(jacobian))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (idx_step + 1) % 100 == 0:
        print("idx_step: {}, loss: {:.2f}".format(idx_step, loss.item()))
    
input_radius = torch.empty(10000, dtype = torch.double, requires_grad = True)
nn.init.uniform_(input_radius, 0, 2*math.pi)
output_radius, jacobian = mt(input_radius)
output_radius = output_radius.detach().numpy()

radius_vm = stats.vonmises.rvs(kappa = kappa, loc = loc, size = 10000)
radius_vm[radius_vm < 0] = radius_vm[radius_vm < 0] + 2*math.pi
radius_vm[radius_vm > 2*math.pi] = radius_vm[radius_vm > 2*math.pi] - 2*math.pi


fig = plt.figure(0)
fig.clf()
plt.hist(output_radius, 40, range = (0, 2*math.pi), alpha = 0.5, label = "samples")
plt.hist(radius_vm, 40, range = (0, 2*math.pi), alpha = 0.5, label = "reference")
plt.legend()
plt.savefig("./output/radius_mobius_transform.pdf")


    
