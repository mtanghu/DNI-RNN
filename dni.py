import torch
import torch.nn as nn
from torch.autograd import Variable


# TODO: add help section for this!! -- maybe make a doc for this somewhere? like in a src file?
# TODO: assert somewhere the hidden state needs to have requires_grad=True, and explain why!
# TODO: figure out how to make an assert that will stop people from forgetting to set retain_grad to true
    # TODO: add assert message to the prev.grad is not None that is helpful

# TODO: ADD A COMMENT ABOUT THE ACCUMULATED GRADIENTS!
# TODO: think about and find out what happens with stacked GRUs and LSTMs



class Synthesizer(nn.Module):
    def __init__(self, size, is_lstm = True, hidden_layers = 1, factor = 1, allow_backwarding = False, activation = nn.GELU,
                 optimizer = torch.optim.Adam, lr = .0001, aux = True, device = 'cuda', use_improvement = True):
        super().__init__()
        self.factor = factor
        self.device = device
        self.is_lstm = is_lstm
        self.use_improvement = use_improvement
        self.allow_backwarding = allow_backwarding
        
        if self.is_lstm:
            # the hidden state and cell state will be concatenated (as per paper) thus doubling gradient size
            size = size * 2
        
        layers = [nn.Linear(size, size)] + [nn.Sequential(activation(), nn.Linear(size, size)) for i in range(hidden_layers)]
        self.synthesizer = nn.Sequential(*layers).to(self.device)
        
        # initialize last layer of synthesizer to only output 0 (as per paper)
        # this will stop spurious gradient from being backwarded at the start
        last_layer = self.synthesizer[-1][-1]
        last_layer.weight.data.fill_(0)
        last_layer.bias.data.fill_(0)
        
        # create auxiliary layers
        self.aux = True
        if self.aux:
            aux_layers = [nn.Linear(size, size)] + [nn.Sequential(activation(), nn.Linear(size, size)) for i in range(hidden_layers)]
            self.aux_layers = nn.Sequential(*aux_layers).to(self.device)

        # somehow this parameters call works
        self.optimizer = optimizer(self.parameters(), lr = lr)
        self.loss_func = nn.MSELoss()
        
        # Variables for future use
        self.predicted_grad = None
        self.prev_hidden = None
        self.prev_synth = None
        self.synthetic_loss = 0


    def backward_synthetic(self, last_hidden):
#         assert self.prev_hidden is None or self.prev_hidden.grad is not None, "make sure to set retain_graph=True for backward call"
        
        last_hidden = torch.cat(last_hidden, dim = 2) if self.is_lstm else last_hidden
        
        # predict future losses, not detach will allow losses from synthetic gradient predict to flow into the model
        if self.allow_backwarding:
            synthetic_gradient = self.synthesizer(last_hidden)
        else:
            synthetic_gradient = self.synthesizer(last_hidden.detach())

        # backward this future loss scaled by a factor (.1 in the paper) for stable training
        last_hidden.backward(gradient = synthetic_gradient.detach() * self.factor, retain_graph = True)
        
        if self.aux and self.predicted_grad is not None:
            aux_loss = self.loss_func(self.predicted_grad, synthetic_gradient.detach())
            aux_loss.backward(retain_graph = True)
            self.aux_loss = aux_loss.item()
            self.predicted_grad = self.aux_layers(last_hidden.detach())

        if self.prev_hidden is not None:
            assert self.prev_hidden.grad is not None
            
            # update synthesizer
            # TODO: ADD A COMMENT ABOUT THE GRADIENTS ACCUMULATED HERE!
            synth_loss = self.loss_func(self.prev_synth, self.prev_hidden.grad)
            synth_loss.backward()

            # store the synthetic gradient loss for monitoring purposes
            self.synthetic_loss = synth_loss.item()

        # save the last hidden state and unroll an extra RNN core by requiring grad (in paper)
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
            # set the last future loss to 0 since this is the end of the epoch
            # this is very important for stopping synthetic gradients from exploding
            synth_loss = self.loss_func(self.prev_synth, torch.zeros(self.prev_synth.shape).to(self.device))
            synth_loss.backward()
        
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        # reset these variables at end of training example
        self.predicted_grad = None
        self.prev_hidden = None
        self.prev_synth = None