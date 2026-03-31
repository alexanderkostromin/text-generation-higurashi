import torch
import pytest
from scripts.tokenizers import HandmadeCharTokenizer
from scripts.datasets import HigurashiDataset


@pytest.fixture
def mock_data_file(tmp_path):
    content = "Кеичи: Привет!\n\n\n\nРена: Ояширо-сама!"
    d = tmp_path / "data"
    d.mkdir()
    f = d / "test_subtitles.txt"
    f.write_text(content, encoding='utf-8')
    return f


def test_higurashi_dataset_logic(mock_data_file):
    context_length = 5
    tokenizer = HandmadeCharTokenizer("Кеичи: Привет! Рена: Ояширо-сама!\n")

    dataset = HigurashiDataset(
        file_path=str(mock_data_file),
        tokenizer=tokenizer,
        context_length=context_length
    )

    assert len(dataset) > 0

    x, y = dataset[0]
    assert x.shape == (context_length,)
    assert y.shape == (context_length,)
    assert x.dtype == torch.long
    assert y.dtype == torch.long

    assert torch.equal(x[1:], y[:-1])

    eos_id = tokenizer.char2idx['<EOS>']
    assert eos_id in dataset.data


def test_higurashi_dataset_batch_integration(mock_data_file):
    tokenizer = HandmadeCharTokenizer("Кеичи: Привет! Рена: Ояширо-сама!\n")
    context_length = 4
    batch_size = 2

    dataset = HigurashiDataset(str(mock_data_file), tokenizer, context_length)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    x_batch, y_batch = next(iter(loader))

    assert x_batch.shape == (batch_size, context_length)
    assert y_batch.shape == (batch_size, context_length)
