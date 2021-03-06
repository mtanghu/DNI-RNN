# Decoupled Neural Interfaces for RNNs in pytorch

This is a tiny library based on [Decoupled Neural Interface using Synthetic Gradients](https://arxiv.org/abs/1608.05343) specifically the part on using synthetic gradients for RNNs. After extensive testing I was able to make some minor improvements that seem to have a significant effect on training stability (which the authors noted to be an issue) as well as increasing the effectiveness of the synthetic gradients (explained later).

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install

```bash
git clone https://github.com/mtanghu/DNI-RNN.git
cd DNI-RNN
pip install .
```

## Why use this package

1. Seemingly this is the only Decoupled Neural Interface (DNI) library that fully implements DNI with RNNs according to the paper (ie. with bootstrapped gradients and the auxiliary task)
2. There is a `use_improvement` parameter that greatly improves the performance DNI
3. We care about UI! There are many checks, warnings, and assertions with helpful explanations to make sure this library is working as intended.


## Usage

You can add DNI to your existing RNN models with ***ONLY 3 MORE LINES*** (not including the import). Let's break down how this works:

### Step 1:
Start by creating a synthesizer for your model passing the hidden size as well as if you're using an LSTM (since an lstm has both a hidden state and a cell state which each need their own gradients). NOTE: you should be able to apply this package to all kinds of RNNs like ones that build on top of LSTMs (ie. with embeddings, weight typing etc.) this can be seen in `examples/copy_task.py`

```python
import dni
synthesizer = dni.Synthesizer(size = MODEL_SIZE, is_lstm = True).to('cpu')
```

### Step 2:
The next step happens within your training loop. After calculating the loss for your model pass the last hidden state to the synthesizer. The synthesizer will backward a synthetic gradient (corresponding to losses from the future). We need to also pass back the hidden state which is detached (to save memory and computation time) but also has `retain_grad=True` to allow future gradients to unroll backwards to the hidden state (normally they wouldn't). __MAKE SURE TO RUN THIS AFTER `loss.backward(retain_graph=True)` AND BEFORE `optimizer.step()`__ we need to retain the graph so the synthesizer can use it (don't worry the synthesizer will free it).

```python
# INSIDE TRAINING LOOP
    hidden_state = synth.backward_synthetic(hidden_state)
```

### Step 3:
Lastly, after you're done with the training example/batch, make sure to update the synthesizer so that it will make better synthetic gradient predictions for the next batch.

```python
# After last input in batch goes through the RNN
synth.step()
```

### Basic example with LSTM and existing TBPTT training loop:
Here is an example of a basic pytorch training loop that hopefully mirrors what you have already. Added lines are denoted by `# NEW LINE HERE`. Note that this code won't actually run and is just meant to show where new code should be added. If you'd like working examples see `examples/`.

```python
# NEW LINE HERE: remember to import package
import dni

MODEL_SIZE = 10
TBPTT = 3
BATCH_SIZE = 16

rnn = nn.LSTM(input_size=MODEL_SIZE, hidden_size=MODEL_SIZE)

# NEW LINE HERE (1): instantiate DNI model
synth = dni.Synthesizer(size = MODEL_SIZE, is_lstm = True).to('cpu')

for X, targets in dataloader:
    hn = (torch.ones(1, BATCH_SIZE, MODEL_SIZE), torch.ones(1, BATCH_SIZE, MODEL_SIZE))
    
    # split training example into TBPTT size sections
    for split in torch.split(X, TBPTT, dim = 1):
        out, hn = rnn(split, hn)
        loss = loss_func(out, targets)
        loss.backward(retain_graph = True)
        
        # NEW LINE HERE (2): backward a synthetic gradient along side the loss gradient
        hn = synth.backward_synthetic(hn)
        
        optim.step()
        optim.zero_grad()
    
    # NEW LINE HERE (3): finish the training example by updating the synthesizer
    synth.step()
```

### Alternative example:
In case you use an RNN where you feed in inputs one at a time like an LSTMCell and then use an if statement to truncate the BPTT, this example should make sense to you.

```python
# NEW LINE HERE: remember to import package
import dni

TBPTT = 5

rnn_cell = nn.LSTMCell(input_size=MODEL_SIZE, hidden_size=MODEL_SIZE)

# NEW LINE HERE (1): instantiate DNI model
synth = dni.Synthesizer(size = MODEL_SIZE, is_lstm = True).to('cpu')

hn = (torch.ones(1, BATCH_SIZE, MODEL_SIZE), torch.ones(1, BATCH_SIZE, MODEL_SIZE))

counter = 0
losses = 0
for X, y in dataloader:
    out, hn = rnn_cell(X, hn)
    losses += loss_func(out, y)
    
    if counter == TBPTT - 1:
        losses.backward(retain_graph = True)
    
        # NEW LINE HERE (2): backward a synthetic gradient along side the loss gradient
        hn = synth.backward_synthetic(h_n, losses)
        
        optim.step()
        optim.zero_grad()

        # NEW LINE HERE (3): finish the training example by updating the synthesizer
        synth.step()
        
        losses = 0
        counter = 0

    counter += 1
```


## Contributing
Contributing is welcome! I'd love to turn this into THE package for Decoupled Neural Interfaces.

Given that this package already implements improvements over the original paper, there's no reason to only implement ideas in the paper.

If you'd like to contribute make sure to install with the -e flag so that edits will be loaded

```bash
git clone https://github.com/mtanghu/DNI-RNN.git
cd DNI-RNN
pip install -e .
```

## License
[MIT](https://choosealicense.com/licenses/mit/)
