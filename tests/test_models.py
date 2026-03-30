import torch
import pytest
from scripts.models import (HandmadeRNN, HandmadeLSTM, HandmadeCasualAttention,
                            HandmadeMutiHeadAttention, HandmadeLayerNorm, HandmadeFeedForward)


@pytest.mark.parametrize('vocab_size, hidden_size, emb_dim, batch_size, seq_len', [
    (100, 32, 16, 4, 10)
])
def test_handmadernn(vocab_size, hidden_size, emb_dim, batch_size, seq_len):
    model = HandmadeRNN(vocab_size, hidden_size, emb_dim)
    test_input = torch.randint(0, 100, (batch_size, seq_len))
    logits, last_h = model(test_input)
    assert logits.shape == (batch_size, seq_len, vocab_size)
    assert last_h.shape == (batch_size, hidden_size)
    assert not torch.isnan(logits).any()


@pytest.mark.parametrize('vocab_size, hidden_size, emb_dim, batch_size, seq_len', [
    (100, 32, 16, 4, 10)
])
def test_handmadelstm(vocab_size, hidden_size, emb_dim, batch_size, seq_len):
    model = HandmadeLSTM(vocab_size, hidden_size, emb_dim)
    test_input = torch.randint(0, 100, (batch_size, seq_len))
    logits, (last_h, last_c) = model(test_input)
    assert logits.shape == (batch_size, seq_len, vocab_size)
    assert last_h.shape == (batch_size, hidden_size)
    assert last_c.shape == (batch_size, hidden_size)
    assert not torch.isnan(logits).any()


@pytest.mark.parametrize('d_in, d_out, context_len, seq_len, batch_size', [
    (3, 2, 12, 10, 2)
])
def test_handmadeca(d_in, d_out, context_len, seq_len, batch_size):
    ca = HandmadeCasualAttention(d_in, d_out, context_len)
    x = torch.randint(0, 100, (batch_size, seq_len, d_in)).float()
    context_vec = ca(x)
    assert context_vec.shape == (batch_size, seq_len, d_out)


@pytest.mark.parametrize('d_in, d_out, context_len, seq_len, batch_size, num_heads', [
    (3, 2, 12, 10, 2, 5)
])
def test_handmademha(d_in, d_out, context_len, seq_len, batch_size, num_heads):
    mha = HandmadeMutiHeadAttention(d_in, d_out, context_len, num_heads)
    x = torch.randint(0, 100, (batch_size, seq_len, d_in)).float()
    context_vec = mha(x)
    assert context_vec.shape == (batch_size, seq_len, d_in)


@pytest.mark.parametrize('batch_size, seq_len, emb_dim, mean, var', [
    (2, 10, 128, 50, 10)
])
def test_handmadeln(batch_size, seq_len, emb_dim, mean, var):
    ln = HandmadeLayerNorm(emb_dim)
    x = torch.randn(batch_size, seq_len, emb_dim) * var + mean
    norm_x = ln(x)

    assert norm_x.shape == (batch_size, seq_len, emb_dim)

    mean = norm_x.mean(dim=-1)
    var = norm_x.var(dim=-1, unbiased=False)

    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)

    assert torch.allclose(var, torch.ones_like(var), atol=1e-5)


@pytest.mark.parametrize('batch_size, seq_len, d_in', [
    (2, 10, 5)
])
def test_handmadeff(batch_size, seq_len, d_in):
    ff = HandmadeFeedForward(d_in)
    x = torch.randn(batch_size, seq_len, d_in)
    res = ff(x)
    assert res.shape == (batch_size, seq_len, d_in)
