import torch
import torch.nn as nn
from torch.autograd import Variable

import traceback


# TODO: add help section for this!! -- maybe make a doc for this somewhere? like in a src file?
    # just somewhere the parameters need to be explained
# TODO: think about and find out what happens with stacked GRUs and LSTMs

# TODO: add last few try catch blocks with the synthetic losses to make sure the .step() is used correctly
# TODO: maybe make a second synthesizer class called SynthesizerZero() which just returns the synthesizier but with allow backwarding flag and aux = False 



class Synthesizer(nn.Module):
    def __init__(self, size, is_lstm = True, hidden_layers = 1, factor = 1, allow_backwarding = False, activation = nn.GELU,
                 optimizer = torch.optim.Adam, lr = .0001, aux = False, use_improvement = True):
        super().__init__()
        self.factor = factor
        self.is_lstm = is_lstm
        self.use_improvement = use_improvement
        self.allow_backwarding = allow_backwarding
        self.device = 'cpu'
        
        if self.is_lstm:
            # the hidden state and cell state will be concatenated (as per paper) thus doubling gradient size
            size = size * 2
        
        layers = [nn.Linear(size, size)] + [nn.Sequential(activation(), nn.Linear(size, size)) for i in range(hidden_layers)]
        self.synthesizer = nn.Sequential(*layers)
        
        # initialize last layer of synthesizer to only output 0 (as per paper)
        # this will stop spurious gradient from being backwarded at the start
        last_layer = self.synthesizer[-1][-1]
        last_layer.weight.data.fill_(0)
        last_layer.bias.data.fill_(0)
        
        # create auxiliary layers
        self.aux = True
        if self.aux:
            #aux_layers = [nn.Linear(size, size)] + [nn.Sequential(activation(), nn.Linear(size, size)) for i in range(hidden_layers)]
            aux_layers = [nn.Linear(size, size)]
            self.aux_layers = nn.Sequential(*aux_layers)

        # somehow this parameters call works
        self.optimizer = optimizer(self.parameters(), lr = lr)
        self.loss_func = nn.MSELoss()
        
        # Variables for future use
        self.predicted_grad = None
        self.prev_hidden = None
        self.prev_synth = None
        self.synthetic_loss = 0
        
    def to(self, device):
        self.device = device
        return super().to(device)
    
    def cuda(self):
        self.device = torch.cuda.current_device()
        return super().cuda()
    
    def cpu(sef):
        self.device = 'cpu'
        return super.cpu()
        
    def backward_synthetic(self, last_hidden):        
        last_hidden = torch.cat(last_hidden, dim = 2) if self.is_lstm else last_hidden
        
        # predict future losses, not detach will allow losses from synthetic gradient predict to flow into the model
        if self.allow_backwarding:
            synthetic_gradient = self.synthesizer(last_hidden)
        else:
            synthetic_gradient = self.synthesizer(last_hidden.detach())
            
        if self.prev_hidden is not None and self.prev_hidden.grad is None:
            raise ValueError(
                "Loss gradient not found, make sure to run .backward_synthetic() AFTER loss.backward(retain_graph=True) and BEFORE optimizer.step()"
                "The graph needs to be retained since .backward_synthetic() uses it. "
                "This error can also happen if you detach your hidden state after calling .backward_synthetic(). Delete the detach line or put it before""
                )

        try:
            # backward this future loss scaled by a factor (.1 in the paper) for stable training
            last_hidden.backward(gradient = synthetic_gradient.detach() * self.factor, retain_graph = True)
        except RuntimeError:
            traceback.print_exc()
            raise RuntimeError(
                'Unable to backward synthetic gradient. See error above, if it says '
                '\"Trying to backward through the graph a second time\" '
                'then you need to set retain_graph=True in your backward call. '
                'Ex: loss.backward(retain_graph=True) the synthesizer uses the graph and will free it on its own. '
                )
        
        # auxilliary task described in paper
        if self.aux and self.predicted_grad is not None:
            aux_loss = self.loss_func(self.predicted_grad, synthetic_gradient.detach())
            aux_loss.backward(retain_graph = True)
            self.aux_loss = aux_loss.item()
            self.predicted_grad = self.aux_layers(last_hidden.detach())

        if self.prev_hidden is not None:
            # update synthesizer with the accumulated gradients in the prev_hidden
            # right now prev_hidden.grad = d_loss/d_prev_hidden + d_future_loss/d_prev_hidden (from the synthetic gradient)
            synth_loss = self.loss_func(self.prev_synth, self.prev_hidden.grad)
            synth_loss.backward()

            # store the synthetic gradient loss for monitoring purposes
            self.synthetic_loss = synth_loss.item()

        # save the last hidden state and unroll an extra RNN core by requiring grad (in paper)
        # graph should be discarded here after the detach unless prev_synth keeps a copy of the graph
        self.prev_hidden = last_hidden.detach()
        self.prev_hidden = Variable(self.prev_hidden, requires_grad = True)
        
        self.prev_synth = synthetic_gradient
        
        if self.is_lstm:
            h, c = torch.chunk(self.prev_hidden, 2, dim = 2)
            return (h.contiguous(), c.contiguous())
        
        return self.prev_hidden
    
    
    def step(self):
        if self.use_improvement:
            # NEW IDEA!!
            # set the last future loss to 0 since this is the end of the epoch to stop synthetic gradients from exploding
            synth_loss = self.loss_func(self.prev_synth, torch.zeros(self.prev_synth.shape).to(self.device))
            synth_loss.backward()
        
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        # reset these variables at end of training example
        self.predicted_grad = None
        self.prev_hidden = None
        self.prev_synth = None
