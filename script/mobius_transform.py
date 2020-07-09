__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2020/05/19 00:59:34"

import numpy as np
import torch
torch.set_default_dtype(torch.double)
import torch.nn as nn
import torch.autograd as autograd
import math
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from sys import exit

def radius_to_xy(radius):
    xy = torch.stack( [torch.cos(radius), torch.sin(radius)], -1)
    return xy

def xy_to_radius(xy):
    flag = xy[:,1] >= 0
    radius = torch.acos(xy[:,0])
    out = flag*radius + (~flag)*(2*math.pi - radius)
    return out

class mobius_transform(nn.Module):
    def __init__(self, w):
        super(mobius_transform, self).__init__()
        self.w = nn.Parameter(w)
        
    def forward(self, radius):
        ## compute the result of the mobius transformation of 0
        ## so that we can use it as the angle of ratation operation
        w_p = 0.99/(1. + torch.norm(self.w))*self.w
        xy = radius_to_xy(w_p.new_zeros(1))
        norm = torch.norm(xy-w_p, dim = -1, keepdim = True)
        out = ((1.-torch.norm(w_p)**2)/norm**2)*(xy-w_p) - w_p
        out = xy_to_radius(out)
        rotate_angle = -out

        ## compute the mobius transformation
        w_p = 0.99/(1. + torch.norm(self.w))*self.w
        xy = radius_to_xy(radius)
        norm = torch.norm(xy-w_p, dim = -1, keepdim = True)
        out = ((1.-torch.norm(w_p)**2)/norm**2)*(xy-w_p) - w_p
        out = xy_to_radius(out)

        ## rotate
        out = out + rotate_angle
        out = torch.remainder(out, 2*math.pi)

        ## compute jacobian
        if radius.requires_grad:
            jacobian = autograd.grad(out, radius,
                                     grad_outputs = torch.ones_like(out),
                                     create_graph = True)[0]
            
        return out, jacobian

    def reverse(self, radius):
        ## compute the result of the mobius transformation of 0
        ## so that we can use it as the angle of ratation operation
        w_p = 0.99/(1. + torch.norm(self.w))*self.w
        xy = radius_to_xy(w_p.new_zeros(1))
        norm = torch.norm(xy-w_p, dim = -1, keepdim = True)
        out = ((1.-torch.norm(w_p)**2)/norm**2)*(xy-w_p) - w_p
        out = xy_to_radius(out)
        rotate_angle = -out

        ## reverse rotation
        radius = radius - rotate_angle
        radius = torch.remainder(radius, 2*math.pi)

        ## reverse the mobius transformation
        w_p = 0.99/(1. + torch.norm(self.w))*self.w
        xy = radius_to_xy(radius)
        norm = torch.norm(w_p + xy, dim=-1, keepdim=True)
        out = w_p + (w_p + xy)*(1. - torch.norm(w_p)**2)/norm**2
        out = xy_to_radius(out)

        ## compute jacobian
        if radius.requires_grad:
            jacobian = autograd.grad(out, radius,
                                     grad_outputs = torch.ones_like(out),
                                     create_graph = True)[0]
            
        return out, jacobian
