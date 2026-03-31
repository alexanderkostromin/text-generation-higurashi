import torch
from torch.utils.data import Dataset


class HigurashiDataset(Dataset):
    def __init__(self, file_path, tokenizer, context_length):
        self.tokenizer = tokenizer
        self.context_length = context_length

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        content = content.replace('\n\n\n\n', '<EOS>')

        encoded_data = tokenizer.encode(content)
        self.data = torch.tensor(encoded_data, dtype=torch.long)

        print(f'Датасет готов. Токенов: {len(self.data)}')

    def __len__(self):
        return len(self.data) - self.context_length - 1

    def __getitem__(self, index):
        x = self.data[index:index + self.context_length]
        y = self.data[index + 1: index + self.context_length + 1]

        return x, y
