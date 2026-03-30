import torch
import pytest
from scripts.models import HandmadeRNN


@pytest.mark.parametrize('vocab_size, hidden_size, emb_dim, batch_size, seq_len, result', [
    (100, 32, 16, 4, 10, [4, 10, 100])
])
def test_handmadernn(vocab_size, hidden_size, emb_dim, batch_size, seq_len, result):
    model = HandmadeRNN(vocab_size, hidden_size, emb_dim)
    test_input = torch.randint(0, 100, (batch_size, seq_len))
    logits, last_h = model(test_input)
    assert logits.shape == (batch_size, seq_len, vocab_size)
    assert last_h.shape == (batch_size, hidden_size)
    assert not torch.isnan(logits).any()
