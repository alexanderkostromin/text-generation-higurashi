import torch
import pytest
from scripts.models import HandmadeRNN, HandmadeLSTM


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
