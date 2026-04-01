import re
from collections import Counter


class HandmadeCharTokenizer:
    def __init__(self, text):
        self.special_tokens = ['<PAD>', '<UNK>', '<EOS>']

        chars_in_text = set(text)
        unique_chars = sorted(list(chars_in_text))

        self.chars = self.special_tokens + unique_chars
        self.char2idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx2char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

        self.spec_reg = re.compile('|'.join(re.escape(t) for t in self.special_tokens))

    def encode(self, s):
        tokens = []
        last_end = 0
        for match in self.spec_reg.finditer(s):
            tokens.extend(list(s[last_end:match.start()]))
            tokens.append(match.group())
            last_end = match.end()
        tokens.extend(list(s[last_end:]))

        return [self.char2idx.get(t, self.char2idx['<UNK>']) for t in tokens]

    def decode(self, indices):
        return ''.join([self.idx2char.get(i, '<UNK>') for i in indices])


class HandmadeWordTokenizer:
    def __init__(self, max_vocab_size=10000, min_freq=2):
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq

        self.special_tokens = ['<PAD>', '<UNK>', '<EOS>']
        self.word2idx = {word: i for i, word in enumerate(self.special_tokens)}
        self.idx2word = {i: word for word, i in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)

    def _tokenize(self, text):
        tokens = re.findall(r'\w+|\s+|[^\w\s]', text)
        return tokens

    def build_vocab(self, text):
        tokens = self._tokenize(text)
        words_frequency = Counter(tokens)

        for word, freq in words_frequency.most_common():
            if freq >= self.min_freq and len(self.word2idx) < self.max_vocab_size:
                if word not in self.word2idx:
                    idx = len(self.word2idx)
                    self.word2idx[word] = idx
                    self.idx2word[idx] = word

        self.vocab_size = len(self.word2idx)

    def encode(self, text, add_special=True):
        tokens = self._tokenize(text)
        indices = []

        for token in tokens:
            idx = self.word2idx.get(token, self.word2idx['<UNK>'])
            indices.append(idx)

        if add_special:
            indices.append(self.word2idx['<EOS>'])

        return indices

    def decode(self, indices):
        words = [self.idx2word.get(idx, '<UNK>') for idx in indices]
        return "".join(words)
