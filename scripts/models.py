import torch
import torch.nn as nn


class HandmadeRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, emb_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.emb = nn.Embedding(vocab_size, emb_dim)

        self.Wxh = nn.Parameter(torch.empty(emb_dim, hidden_size))
        self.Whh = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.Why = nn.Parameter(torch.empty(hidden_size, vocab_size))
        nn.init.xavier_uniform_(self.Wxh)
        nn.init.xavier_uniform_(self.Whh)
        nn.init.xavier_uniform_(self.Why)

        self.bh = nn.Parameter(torch.zeros(hidden_size))
        self.by = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, x, hidden=None):
        emb = self.emb(x)   # [B, S, E]

        batch_size, seq_len, _ = emb.size()

        if hidden is None:
            hidden = torch.zeros(batch_size, self.hidden_size).to(x.device)

        outputs = []

        for t in range(seq_len):
            x_t = emb[:, t, :]  # [B, E]
            hidden = torch.tanh(x_t @ self.Wxh + hidden @ self.Whh + self.bh)
            outputs.append(hidden.unsqueeze(1))

        full_hidden_states = torch.cat(outputs, dim=1)

        logits = full_hidden_states @ self.Why + self.by

        return logits, hidden


class HandmadeLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size, emb_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.emb = nn.Embedding(vocab_size, emb_dim)

        self.Wxh = nn.Parameter(torch.empty(emb_dim, 4 * hidden_size))
        self.Whh = nn.Parameter(torch.empty(hidden_size, 4 * hidden_size))
        self.Why = nn.Parameter(torch.empty(hidden_size, vocab_size))
        nn.init.xavier_uniform_(self.Wxh)
        nn.init.xavier_uniform_(self.Whh)
        nn.init.xavier_uniform_(self.Why)

        self.bh = nn.Parameter(torch.zeros(4 * hidden_size))
        self.by = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, x, init_states=None):
        emb = self.emb(x)   # [B, S, E]
        batch_size, seq_len, _ = emb.size()

        if init_states is None:
            h_t = torch.zeros(batch_size, self.hidden_size).to(x.device)
            c_t = torch.zeros(batch_size, self.hidden_size).to(x.device)
        else:
            h_t, c_t = init_states

        outputs = []

        for t in range(seq_len):
            x_t = emb[:, t, :]

            gates = x_t @ self.Wxh + h_t @ self.Whh + self.bh

            i_gate, f_gate, g_gate, o_gate = gates.chunk(4, dim=1)

            i = torch.sigmoid(i_gate)    # Input gate
            f = torch.sigmoid(f_gate)    # Forget gate
            o = torch.sigmoid(o_gate)    # Output gate
            g = torch.tanh(g_gate)       # cell state

            c_t = f * c_t + i * g

            h_t = o * torch.tanh(c_t)

            outputs.append(h_t.unsqueeze(1))

        full_output = torch.cat(outputs, dim=1)
        logits = full_output @ self.Why + self.by

        return logits, (h_t, c_t)
