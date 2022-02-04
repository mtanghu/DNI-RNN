import torch
import torch.nn as nn
import dni

BATCH_SIZE = 16
D_MODEL = 256
TBPTT = 3
COPY_SIZE = 3*TBPTT - 1
ALPHABET = "01"
EPOCHS = 10

# TODO: add arg parser
    # make clip grad a param
# TODO: show bits error with more comprehensive output (see pytorch WLM example)


class LSTM_plus_embedding(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.embedding = nn.Embedding(len(ALPHABET)+1, size)
        self.rnn = nn.LSTM(size, hidden_size = size)
        self.output_embedding = nn.Linear(size, len(ALPHABET)+1)
        
        # # weight tying
        # self.output_embedding.weight = self.embedding.weight
       
        
    def forward(self, input, h0):
        x = self.embedding(input)
        out, hn = self.rnn(x, h0)
        out = self.output_embedding(out)
        return out, hn
    

if __name__ == "__main__":
    device = 'cpu'
    
    ### DATA ###
    # create prompts with a stop character at the end
    batched_prompts = torch.randint(0, len(ALPHABET), (EPOCHS, COPY_SIZE, BATCH_SIZE), dtype = torch.long)
    batched_prompts = torch.cat((batched_prompts, len(ALPHABET)*torch.ones(EPOCHS, 1, BATCH_SIZE, dtype = torch.long)), dim=1)
    batched_prompts = torch.utils.data.DataLoader(batched_prompts, batch_size = None, pin_memory = True)
    
    rnn = LSTM_plus_embedding(D_MODEL).to(device)
    optim = torch.optim.SGD(rnn.parameters(), lr = .001)
    loss = nn.CrossEntropyLoss()

    # instantiate DNI model that will backward synthetic gradients
    synth = dni.Synthesizer(size = D_MODEL, is_lstm = True, device = device)
    
    losses = []
    synth_losses = []
    aux_losses = []
    synths = []
    for epoch, batch in enumerate(batched_prompts):
        batch = batch.to(device)
        
        # initialize the hidden state with synth
        h_n = (torch.ones(1, BATCH_SIZE, D_MODEL).to(device), torch.ones(1, BATCH_SIZE, D_MODEL).to(device))

        # split into TBPTT size sections
        for split in torch.split(batch, TBPTT, dim = 0):
            # standard forward pass
            out, h_n = rnn(split, h_n)
            cross_loss = loss(out.view(-1, len(ALPHABET)+1), torch.zeros(BATCH_SIZE, TBPTT, dtype = torch.long).view(-1).to(device))
            
            # just add ONE line for synthetic gradients
            h_n = synth.backward_synthetic(h_n)
            
            cross_loss.backward()

            torch.nn.utils.clip_grad_norm_(rnn.parameters(), 25)
            optim.step()
            optim.zero_grad()

        # check copy ability
        for split in torch.split(batch, TBPTT, dim = 0):
            # standard forward pass
            out, h_n = rnn(torch.zeros(TBPTT, BATCH_SIZE, dtype = torch.long).to(device), h_n)
            cross_loss = loss(out.reshape(-1, len(ALPHABET)+1), split.reshape(-1))
            
            # just add ONE line for synthetic gradients
            h_n = synth.backward_synthetic(h_n)
            
            cross_loss.backward()

            losses.append(cross_loss.item())

            torch.nn.utils.clip_grad_norm_(rnn.parameters(), 25)
            optim.step()
            optim.zero_grad()
        
        # step the the DNI model 
        synth.step()

        print(f'epoch: {epoch}', end = '\r')

