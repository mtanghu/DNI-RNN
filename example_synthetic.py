import torch
import torch.nn as nn
import torch.nn.functional as F


# Paper says to scale synthetic gradients by .1
FACTOR = .1
BATCH_SIZE = 32


# size is the hidden and output size
size = 20
random_input = torch.ones(10, BATCH_SIZE, size)
loss = nn.CrossEntropyLoss()
mse_loss = nn.MSELoss()

# this will create the synthetic gradients (this is M_T in the paper)
synth = nn.Sequential(nn.Linear(size, size), nn.GELU(), nn.Linear(size, size))

# paper says to intialize weights and biases to 0
synth[2].weight.data.fill_(0)
synth[2].bias.data.fill_(0)

# make RNN (ie. the GRU) with optimizers
gru = nn.GRU(size, hidden_size = size)
optim = torch.optim.Adam(gru.parameters())
synth_optim = torch.optim.Adam(synth.parameters())

# run a forward pass
out, h_n = gru(random_input)

# the target in the loss is just 3s (arbitrary)
ce = loss(out, 3*torch.ones(10,20, dtype=torch.long))

# backward the loss, make sure to retain the graph to allow for second backward pass
ce.backward(retain_graph = True)

# also backward the syntheic gradient
synth_grad = synth(h_n.detach())
h_n.backward(gradient = synth_grad.detach()*FACTOR, retain_graph = True)

# AUXILIARY GRADIENTS
aux = synth(out[:-1].detach())

optim.step()
optim.zero_grad()
h_n = h_n.detach()

# keep h_n gradient for future use
h_n.requires_grad = True

# forward pass
out, h_n2 = gru(random_input, h_n)
ce = loss(out, 3*torch.ones(10,20, dtype=torch.long))

# AUXILIARY GRADIENTS
aux2 = synth(out[:-1].detach())

# AUX TASK
aux_loss = mse_loss(aux, aux2)
aux_loss.backward()

# get the true gradient dL/dh_n
true_grad = torch.autograd.grad(outputs = ce, inputs = h_n, retain_graph = True)[0]

# get bootstraped gradient (bottom of page 3)
h_n2.backward(torch.ones(h_n2.shape), inputs = h_n, retain_graph = True)
bootstrap_grad = true_grad + synth_grad.detach() * h_n.grad

# make sure this grad doesn't go anywhere since it's just an intermediate calculation
h_n.grad = None

# update synthesizer
synth_loss = mse_loss(synth_grad, bootstrap_grad.detach())
synth_loss.backward(retain_graph = True)
synth_optim.step()
synth_optim.zero_grad()

ce.backward(retain_graph = True)

# pass the synthetic gradient backwards
synth_grad = synth(h_n2.detach())
h_n2.backward(gradient = synth_grad.detach()*FACTOR, retain_graph = True)

optim.step()
optim.zero_grad()
h_n2 = h_n2.detach()

# TODO: turn this in to a loop!