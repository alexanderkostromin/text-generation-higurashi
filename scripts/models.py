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
