import config
import torchtext
import torch
def dataLoader():
    print('loading data..')
    TEXT = torchtext.data.Field(lower=True)
    train, val, test = torchtext.datasets.LanguageModelingDataset.splits(path="./data", 
        train="text8.train.txt", validation="text8.dev.txt", test="text8.test.txt", text_field=TEXT)
    TEXT.build_vocab(train, max_size=config.MAX_VOCAB_SIZE)
    
    print("vocabulary size: {}".format(len(TEXT.vocab)))

    device = torch.device('cuda' if config.USE_CUDA else 'cpu')

    train_iter, val_iter, test_iter = torchtext.data.BPTTIterator.splits(
        (train, val, test), batch_size=config.BATCH_SIZE, device=device, bptt_len=32, repeat=False, shuffle=True)

    return len(TEXT.vocab), train_iter, test_iter, val_iter


if __name__ == "__main__":
    VOCAB_SIZE, train_iter, test_iter, val_iter = dataLoader()

    it = iter(train_iter)
    batch = next(it)
    x = batch.target.view(-1)
    print(x.shape)