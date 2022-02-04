import torch
import torch.nn as nn
from torch.autograd import Variable

class DNI_RNN(nn.Module):
    def __init__(self, model, hidden_layers = 1, factor = .1, activation = nn.GELU,
                 optimizer = torch.optim.Adam, lr = .0001, aux = True, device = 'cuda'):
        super().__init__()
        self.is_lstm = type(model) == nn.LSTM
        self.factor = factor
        self.device = device
        
        # the hidden state and the cell state will be concatted for LSTMs (as per paper) doubling the gradient size
        size = model.hidden_size * 2 if self.is_lstm else model.hidden_size
        
        layers = [nn.Linear(size, size)] + [nn.Sequential(activation(), nn.Linear(size, size))
                                            for i in range(hidden_layers)]
        self.synthesizer = nn.Sequential(*layers).to_device(self.device)
        
        # initialize last layer of synthesizer to only output 0 (as per paper)
        # this will stop spurious gradient from being backwarded at the start
        last_layer = self.synthesizer[-1][-1]
        last_layer.weight.data.fill_(0)
        last_layer.bias.data.fill_(0)
        
        # create auxiliary
        self.aux = True
        if self.aux:
            self.aux_layers = [nn.Linear(size, size)] + [nn.Sequential(activation(), nn.Linear(size, size))
                                                         for i in range(hidden_layers)]
            self.aux_layers = self.aux_layer.to_device(self.device)

        # somehow this parameters call works
        self.optimizer = optimizer(self.parameters(), lr = lr)
        self.loss_func = nn.MSELoss()
        
        
    def new_hidden(self, hidden):
        self.prev_hidden = torch.cat(hidden, dim = 2) if self.is_lstm else hidden
        
        # make sure hidden state requires grad (ie. unrolling one more core)
        self.prev_hidden = Variable(self.prev_hidden, requires_grad = True)

        self.prev_synth = self.synthesizer(self.prev_hidden.detach())
        
        if self.aux:
            self.predicted_grad = self.aux_layer(self.prev_hidden)
            
        if self.lstm:
            h, c = torch.chunk(self.prev_hidden, 2, dim = 2)
            return (h.contiguous(), c.contiguous())
        
        return self.prev_hidden


    def backward_synthetic(self, last_hidden, loss):
        last_hidden = torch.cat(last_hidden, dim = 2) if self.is_lstm else last_hidden
        
        # predict future losses and backward that gradient
        synthetic_gradient = self.synthesizer(last_hidden.detach())
        last_hidden.backward(gradient = synthetic_gradient.detach() * self.factor, retain_graph = True)
        
        if self.aux:
            aux_loss = self.loss_func(self.predicted_grad, synthetic_gradient.detach())
            aux_loss.backward(retain_graph = True)
            self.aux_loss = aux_loss.item()
            self.predicted_grad = self.aux_layers(last_hidden.detach())

        # get the real loss gradient d_Loss/d_prev_hidden
        self.prev_hidden.grad = None
        real_grads = torch.autograd.grad(outputs = loss, inputs = self.prev_hidden, retain_graph = True)[0]
        self.prev_hidden.grad = None
        
        # get d_future_loss/d_prev_hidden
        last_hidden.backward(gradient = synthetic_gradient.detach(), inputs = self.prev_hidden, retain_graph = True)
        
        # add the above gradients to get the bootstrapped gradient of all future losses w.r.t the prev_hidden
        bootstrap_grad = real_grads + self.prev_hidden.grad

        # make sure this grad doesn't go anywhere since it's just an intermediate calculation
        self.prev_hidden.grad = None

        # update synthesizer
        synth_loss = self.loss_func(self.prev_synth, bootstrap_grad.detach())
        synth_loss.backward()
        self.boot = bootstrap_grad
        
        # store the synthetic gradient loss for monitoring purposes
        self.synthetic_loss = synth_loss.item()

        # save the last hidden state and unroll an extra RNN core by requiring grad (in paper)
        self.prev_hidden = last_hidden.detach()
        self.prev_hidden = Variable(self.prev_hidden, requires_grad = True)
        
        self.prev_synth = synthetic_gradient
        
        if self.lstm:
            h, c = torch.chunk(self.prev_hidden, 2, dim = 2)
            return (h.contiguous(), c.contiguous())
        
        return self.prev_hidden
    
    
    def step(self):
        # NEW IDEA!!
        # set the last future loss to 0 since this is the end of the epoch
        synth_loss = self.loss_func(self.prev_synth, torch.zeros(self.prev_synth.shape).to_device(self.device))
        synth_loss.backward()
        
        self.optimizer.step()
        self.optimizer.zero_grad()
