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
            g = torch.tanh(g_gate)       # Cell state

            c_t = f * c_t + i * g

            h_t = o * torch.tanh(c_t)

            outputs.append(h_t.unsqueeze(1))

        full_output = torch.cat(outputs, dim=1)
        logits = full_output @ self.Why + self.by

        return logits, (h_t, c_t)


class HandmadeCasualAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.context_length = context_length

        self.Wq = nn.Parameter(torch.empty(d_in, d_out))
        self.Wk = nn.Parameter(torch.empty(d_in, d_out))
        self.Wv = nn.Parameter(torch.empty(d_in, d_out))
        nn.init.xavier_uniform_(self.Wq)
        nn.init.xavier_uniform_(self.Wk)
        nn.init.xavier_uniform_(self.Wv)

        self.bq = nn.Parameter(torch.zeros(d_out))
        self.bk = nn.Parameter(torch.zeros(d_out))
        self.bv = nn.Parameter(torch.zeros(d_out))

        self.register_buffer('mask',
                             torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        batch_size, seq_len, d_in = x.size()

        queries = x @ self.Wq + self.bq     # [B, S, d_in] @ [d_in, d_out] = [B, S, d_out]
        keys = x @ self.Wk + self.bk        # [B, S, d_in] @ [d_in, d_out] = [B, S, d_out]
        values = x @ self.Wv + self.bv      # [B, S, d_in] @ [d_in, d_out] = [B, S, d_out]

        attn_scores = queries @ keys.transpose(1, 2)    # [B, S, d_out] @ [B, d_out, S] = [B, S, S]
        attn_scores.masked_fill_(self.mask.bool()[:seq_len, :seq_len], float('-inf'))

        attn_weights = torch.softmax(attn_scores / self.d_out**0.5, dim=-1)

        context_vec = attn_weights @ values     # [B, S, S] @ [B, S, d_out] = [B, S, d_out]

        return context_vec


class HandmadeMutiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, num_heads):
        super().__init__()
        self.heads = nn.ModuleList(
            [HandmadeCasualAttention(d_in, d_out, context_length) for _ in range(num_heads)]
        )
        self.out_projection = nn.Parameter(torch.empty(d_out * num_heads, d_in))
        self.bout = nn.Parameter(torch.zeros(d_in))
        nn.init.xavier_uniform_(self.out_projection)

    def forward(self, x):
        combined = torch.cat([head(x) for head in self.heads], dim=-1)  # [B, S, d_out * num_heads]

        # [B, S, d_out * num_heads] @ [d_out * num_heads, d_in] = [B, S, d_in]
        context_vec = combined @ self.out_projection + self.bout
        return context_vec


class HandmadeLayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift