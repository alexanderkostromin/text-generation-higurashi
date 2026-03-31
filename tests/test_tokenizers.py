import pytest
from scripts.tokenizers import HandmadeCharTokenizer


@pytest.mark.parametrize('text', [
    ('абв')
     ])
def test_hm_char_tokenizer(text):
    tokenizer = HandmadeCharTokenizer(text)
    num_special_tokens = 3
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)

    encoded_unknown = tokenizer.encode('г')

    assert decoded == text

    assert len(tokenizer.char2idx) == len(set(text)) + num_special_tokens

    assert tokenizer.decode(encoded_unknown) == '<UNK>'

    assert text[0] in tokenizer.char2idx
    assert text[1] in tokenizer.char2idx
