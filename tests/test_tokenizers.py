import pytest
from scripts.tokenizers import HandmadeCharTokenizer, HandmadeWordTokenizer


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


@pytest.mark.parametrize('text', [
    ('Внутри меня сидит демон. В имени. В теле. В сердце')
])
def test_hm_word_tokenizer(text):
    tokenizer = HandmadeWordTokenizer(min_freq=1)

    tokenizer.build_vocab(text)

    encoded = tokenizer.encode(text, add_special=False)
    decoded = tokenizer.decode(encoded)

    assert decoded == text

    expected_tokens = set(tokenizer._tokenize(text))
    assert tokenizer.vocab_size == len(expected_tokens) + 3

    encoded_unknown = tokenizer.encode('неизвестное_слово', add_special=False)
    assert tokenizer.decode(encoded_unknown) == '<UNK>'

    tokens = tokenizer._tokenize(text)
    assert tokens[0] in tokenizer.word2idx
    assert tokens[1] in tokenizer.word2idx
