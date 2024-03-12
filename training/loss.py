# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import torch
from torch_utils import persistence

from scipy.optimize import linear_sum_assignment
#import cupy as cp
#function for assignment problem
def assignment_problem(noise, image, labels = None):
    N,C,W,H = image.shape
    dist_matrix = torch.cdist(image.reshape(N,C*W*H),noise.reshape(N,C*W*H))

    #dist_matrix = cp.asarray(dist_matrix)
    dist_matrix_np = dist_matrix.detach().cpu().numpy()
    del dist_matrix
    a,b = linear_sum_assignment(dist_matrix_np)
    del dist_matrix_np
    return noise[b]


def assignment_problem_label(noise, image,labels):
    N,C,W,H = image.shape
    _,L = labels.shape
    labels_idx= [[] for i in range(L)]
    #a_label = [0 for i in range(N)]
    a_label = torch.argmax(labels, 1)

    for i in range(N):
        labels_idx[a_label[i]].append(i)
    
    b_lst=[]
    for i in range(labels.shape[1]):
        dist_matrix = torch.cdist(image.reshape(N,C*W*H)[labels_idx[i]],noise.reshape(N,C*W*H)[labels_idx[i]])

    #dist_matrix = cp.asarray(dist_matrix)
        dist_matrix = dist_matrix.detach().cpu().numpy()

        a,b = linear_sum_assignment(dist_matrix)
        b_lst.append(b)
        #print(b.sort())
    new_lst = [0 for i in range(N)]
    for i in range(N) :
        #print(i)
        new_lst[i] = labels_idx[a_label[i]][b_lst[a_label[i]][0]]
        b_lst[a_label[i]] = b_lst[a_label[i]][1:]
    #print(new_lst)
    return noise[new_lst]


#----------------------------------------------------------------------------
# Loss function corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VPLoss:
    def __init__(self, beta_d=19.9, beta_min=0.1, epsilon_t=1e-5):
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()

#----------------------------------------------------------------------------
# Loss function corresponding to the variance exploding (VE) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VELoss:
    def __init__(self, sigma_min=0.02, sigma_max=100):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

#----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).

@persistence.persistent_class
class EDMLoss_tp:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5,batch = 32):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.batch = batch

    def __call__(self, net, images, labels=None, augment_pipe=None, noise = None):
        N = self.batch
        rnd_normal = torch.randn([N, 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        with torch.no_grad():
            n_ = torch.randn_like(y)
            #n_ = assignment_problem(n_,y)
            n_ = assignment_problem_label(n_,y,labels)
        n = n_[0:N,:,:,:] * sigma
        y_= y[0:N,:,:,:]
        del y, n_
        D_yn = net(y_ + n, sigma, labels[0:N,:], augment_labels=augment_labels[0:N,:])
        loss = weight * ((D_yn - y_) ** 2)
        return loss


@persistence.persistent_class
class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5,batch = 32):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.batch = batch

    def __call__(self, net, images, labels=None, augment_labels=None, noise = None):
        N = self.batch
        rnd_normal = torch.randn([N, 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        #y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        y = images
        n =  noise * sigma
        
        
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

#----------------------------------------------------------------------------
