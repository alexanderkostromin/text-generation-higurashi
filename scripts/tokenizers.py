import re


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
