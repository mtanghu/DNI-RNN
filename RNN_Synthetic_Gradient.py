import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt


# Paper says to scale synthetic gradients by .1
FACTOR = .1
BATCH_SIZE = 32


# size is the hidden and output size
size = 20
random_input = torch.ones(10, BATCH_SIZE, dtype = torch.long)
loss = nn.CrossEntropyLoss()
mse_loss = nn.MSELoss()

# this will create the synthetic gradients (this is M_T in the paper)
synth = nn.Sequential(nn.Linear(size, size), nn.GELU(), nn.Linear(size, size))

# paper says to intialize weights and biases to 0
synth[2].weight.data.fill_(0)
synth[2].bias.data.fill_(0)

class GRU_Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.embedding(size,size)
        self.gru = nn.GRU(size, hidden_size = size)
        self.linear_layer = nn.Linear(size, size)
        self.embedding.weight = self.linear_layer.weight
       
        
    def forward(self, input, h0):
        x = self.embedding(input)
        out, hn = self.gru(x, h0)
        out = self.linear_layer(out)
        return out, hn

# make RNN (ie. the GRU) with optimizers
gru = GRU_Layer()
optim = torch.optim.Adam(gru.parameters())
synth_optim = torch.optim.Adam(synth.parameters())

h_n_prev = torch.ones(1, BATCH_SIZE, size, requires_grad=True)

ce_loss_list = []
aux_loss_list = []
synth_loss_list = []

for i in range(100):
    # AUXILIARY GRADIENTS
    aux_future_pred = synth(h_n_prev.detach())

    # run a forward pass
    out, h_n = gru(random_input, h_n_prev)
    
    # the target in the loss is just 3s (arbitrary)
    ce = loss(out.view(-1, size), torch.zeros(10, 32, dtype = torch.long).view(-1))
    
    ce_loss_list.append(ce)

    # backward the loss, make sure to retain the graph to allow for second backward pass
    ce.backward(retain_graph = True)

    # also backward the syntheic gradient
    synth_grad = synth(h_n.detach())
    h_n.backward(gradient = synth_grad.detach()*FACTOR, retain_graph = True)

    # AUX PREDICT FUTURE GRADIENT
    aux_loss = mse_loss(aux_future_pred, synth_grad.detach())
    aux_loss_list.append(aux_loss)
    aux_loss.backward(retain_graph=True)

    # get the true gradient dL/dh_n
    true_grad = torch.autograd.grad(outputs = ce, inputs = h_n_prev, retain_graph = True)[0]

    # get bootstraped gradient (bottom of page 3)
    h_n.backward(torch.ones(h_n.shape), inputs = h_n_prev, retain_graph = True)
    bootstrap_grad = true_grad + synth_grad.detach() * h_n_prev.grad

    # make sure this grad doesn't go anywhere since it's just an intermediate calculation
    h_n_prev.grad = None

    # update synthesizer
    synth_loss = mse_loss(synth_grad, bootstrap_grad.detach())
    synth_loss_list.append(synth_loss)
    synth_loss.backward(retain_graph = True)
    synth_optim.step()
    synth_optim.zero_grad()

    optim.step()
    optim.zero_grad()
    h_n = h_n.detach()

    # keep h_n gradient for future use
    h_n.requires_grad = True
    h_n_prev = h_n

plt.plot(range(100), ce_loss_list)
plt.title('Cross entropy loss per epoch')
plt.show()

plt.plot(range(100), aux_loss_list)
plt.title('Auxiliary loss per epoch')
plt.show()

plt.plot(range(100), synth_loss_list)
plt.title('Synthetic gradient loss per epoch')
plt.show()

