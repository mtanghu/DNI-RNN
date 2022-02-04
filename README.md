# Decoupled Neural Interfaces for RNNs in pytorch

This is a tiny library based on [Decoupled Neural Interface using Synthetic Gradients](https://arxiv.org/abs/1608.05343) specifically the part on using synthetic gradients for RNNs. After extensive testing I was able to make some minor improvements that seem to have a significant effect on training stability (which the authors noted to be an issue) as well as increasing the effectiveness of the synthetic gradients (explained later).

## TODO: EXPLAIN CONCEPT WITH DIAGRAM, THEN EXPLAIN MATH AND IMPROVEMENT
## TODO: SHOW A GRAPH WITH TIMING OF SPEED INCREASE, ALSO SHOW MEMORY IMPROVEMENT
## TODO: MAKE THIS FRIENDLY FOR NON PYTORCH USERS, SPECIFICALLY ADD DETAILS ABOUT BPTT AND HOW RNNs WORK IN PYTORCH

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install

```bash
git clone https://github.com/mtanghu/DNI-RNN.git
cd DNI-RNN
pip install .
```

## Usage

You can add DNI to your existing RNN models with ONLY 5 MORE LINES. Here is an example of a basic pytorch training loop with an LSTM with Truncated Backpropgration Through Time (TBPTT). Added lines are denoted by ```# NEW LINE HERE```. Note that this code won't actually run and is just meant to show where new code should be added. If you'd like working examples see `examples/`.

```python
# NEW LINE HERE (1): remember to import package
import dni

MODEL_SIZE = 10
TBPTT = 3
BATCH_SIZE = 16

rnn = nn.LSTM(input_size=MODEL_SIZE, hidden_size=MODEL_SIZE)

# NEW LINE HERE (2): instantiate DNI mode and let it know if you're using an LSTM/the hidden state comes from a LSTM
synth = dni.Synthesizer(size = MODEL_SIZE, is_lstm = True).cuda()

for X, y in dataloader:
    hn = (torch.ones(1, BATCH_SIZE, MODEL_SIZE),
          torch.ones(1, BATCH_SIZE, MODEL_SIZE))
    
    # NEW LINE HERE (3): initialize hidden state with the synthesizer at the start of the training example
    hn = synth.init_hidden(hn)
    
    # split training example into TBPTT size sections
    for split in torch.split(X, TBPTT, dim = 1):
        out, hn = rnn(split, hn)
        loss = loss_func(out, y)
        
        # NEW LINE HERE (4): backward a synthetic gradient along side the loss gradient (note: do before the loss.backward() call))
        hn = synth.backward_synthetic(h_n, cross_loss)
        
        loss.backward()
        optim.step()
        optim.zero_grad()
    
    # NEW LINE HERE (5): finish the training example by updating the synthesizer
    synth.step()
```

Alternative example with LSTMCell (where you feed inputs in one at a time rather than as a sequence) and you use an if statement to truncate the BPTT.

```python
# NEW LINE HERE (1): remember to import package
import dni

TBPTT = 5

rnn = nn.LSTMCell(input_size=MODEL_SIZE, hidden_size=MODEL_SIZE)

# NEW LINE HERE (2): instantiate DNI mode and let it know if you're using an LSTM/the hidden state comes from a LSTM
synth = dni.Synthesizer(size = MODEL_SIZE, is_lstm = True).cuda()

hn = (torch.ones(1, BATCH_SIZE, MODEL_SIZE),
      torch.ones(1, BATCH_SIZE, MODEL_SIZE))

# NEW LINE HERE (3): initialize hidden state with the synthesizer at the start of the training example
hn = synth.init_hidden(hn)

counter = 0
losses = 0
for X, y in dataloader:
    out, hn = rnn(X, hn)
    losses += loss_func(out, y)
    
    if counter == TBPTT:
        # NEW LINE HERE (4): backward a synthetic gradient along side the loss gradient (note: do before the loss.backward() call))
        hn = synth.backward_synthetic(h_n, losses)

        losses.backward()
        optim.step()
        optim.zero_grad()

        # NEW LINE HERE (5): finish the training example by updating the synthesizer
        synth.step()
        counter = 0

    counter += 1
```


## Contributing
Contributing is welcome! I'd love to turn this into THE package for Decoupled Neural Interfaces.

Given that this package already implements improvements over the original paper, there's no reason to only implement ideas in the paper. The paper mentions that synthetic gradients in RNNs is analgous to temporal credit in Reinforcement Learning so I wonder if this package could be used in that direction.

If you'd like to contribute make sure to install with the -e flag so that edits will be loaded

```bash
git clone https://github.com/mtanghu/DNI-RNN.git
cd DNI-RNN
pip install -e .
```

## License
[MIT](https://choosealicense.com/licenses/mit/)
