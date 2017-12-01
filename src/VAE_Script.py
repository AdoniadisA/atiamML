# @pierrotechnique
# -*- coding: utf-8 -*-
#%%
import tensorflow.examples.tutorials.mnist as tetm
mnist = tetm.input_data.read_data_sets('../data/MNIST',one_hot=True)
#? How are tensorflow datasets structured?
#? What does the one_hot option do?

#%%
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

X_dim = mnist.train.images.shape[1] # Input variable (X) dimensionality
#? Why does this correspond to shape[1]?
miniBatchSize = 49
z_dim = 49 # Latent space dimensionality
h_dim = 196 # Hidden layer (h) dimensionality
Nit = 1000
learningRate = 0.001

def varInit(size): # Initializes network input variables
    inpDim = size[0] # Assumes size = [inpDim,outDim]
    stdDev = 1./np.sqrt(inpDim/2.)
    return torch.autograd.Variable(
            torch.randn(size)*stdDev,requires_grad=True)

# Encoding
wXh = varInit(size=[X_dim,h_dim]) # Weights X into h
bXh = torch.autograd.Variable(torch.zeros(h_dim),requires_grad=True) # Bias

whz_mu = varInit(size=[h_dim,z_dim]) # Weights h into z (mu(X))
bhz_mu = torch.autograd.Variable(torch.zeros(z_dim),requires_grad=True)

whz_sigma = varInit(size=[h_dim,z_dim]) # Weights h into z (sigma(X))
bhz_sigma = torch.autograd.Variable(torch.zeros(z_dim),requires_grad=True)

def Q(X): # Two-layer encoder network
    h = torch.nn.functional.relu(torch.mm(X,wXh) + bXh)
    z_mu = torch.mm(h,whz_mu) + bhz_mu
    z_sigma = torch.mm(h,whz_sigma) + bhz_mu
    return z_mu,z_sigma

def zParam(mu,sigma): # z latent variable reparameterization trick
    eps = torch.autograd.Variable(torch.randn(miniBatchSize,z_dim))
#    return mu + torch.exp(torch.log(sigma)/2.)*eps
    return mu + torch.exp(sigma/2.)*eps

# Decoding
wzh = varInit(size=[z_dim,h_dim]) # Weights z into h
bzh = torch.autograd.Variable(torch.zeros(h_dim),requires_grad=True)

whX = varInit(size=[h_dim,X_dim]) # Weights h into X
bhX = torch.autograd.Variable(torch.zeros(X_dim),requires_grad=True)

def P(z): # Two-layer decoder network
    h = torch.nn.functional.relu(torch.mm(z,wzh) + bzh)
    X = torch.nn.functional.sigmoid(torch.mm(h,whX) + bhX)
    return X

# Training
param = [wXh,bXh,whz_mu,bhz_mu,whz_sigma,bhz_sigma,wzh,bzh,whX,bhX]
solver = torch.optim.Adam(param,lr=learningRate)

for it in xrange(Nit):
    X,_ = mnist.train.next_batch(miniBatchSize)
    X = torch.autograd.Variable(torch.from_numpy(X))   
    # Forward
    z_mu,z_sigma = Q(X)
    z = zParam(z_mu,z_sigma)
    Xout = P(z)
    # Loss
#    reconLoss = torch.nn.functional.binary_cross_entropy(
#            Xout,X,size_average=False)/miniBatchSize # wiseodd
    reconLoss = torch.nn.functional.binary_cross_entropy(
            Xout,X) #pytorch
    #? What does size_average do in this function?
#    klLoss = 0.5*torch.sum(torch.exp(z_sigma)+(z_mu**2)-1.-z_sigma) # wiseodd
    klLoss = -0.5*torch.sum(-(z_sigma.exp())-(z_mu.pow(2))+1.+z_sigma)
    klLoss /= (miniBatchSize*X.size()[1]) # pytorch
    loss = reconLoss + klLoss
    # Backward
    loss.backward()
    # Update
    solver.step()
    # Clear parameter gradients (manually)
    for p in param:
        if p.grad is not None:
            data = p.grad.data
            p.grad = torch.autograd.Variable(
                    data.new().resize_as_(data).zero_())
    if ((it % 100) == 0):
        print('Loss: '+str(loss.data[0]))
            
print('Finished training, brah!')

#%%

#samples = X.data.numpy()
samples = P(z).data.numpy()

fig = plt.figure(figsize=(8,8))
gs = gridspec.GridSpec(8,8)
gs.update(wspace=0.1,hspace=0.1)

for i,sample in enumerate(samples):
    ax = plt.subplot(gs[i])
    plt.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')
    plt.imshow(sample.reshape(28,28),cmap='Greys_r')