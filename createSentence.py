
import torch
from RNNModel import RNNModel
import config
import torchtext
if __name__ == "__main__":

    TEXT = torchtext.data.Field(lower=True)
    train, val, test = torchtext.datasets.LanguageModelingDataset.splits(path="./data", 
        train="text8.train.txt", validation="text8.dev.txt", test="text8.test.txt", text_field=TEXT)

    TEXT.build_vocab(train, max_size=config.MAX_VOCAB_SIZE)
    model = RNNModel(50002,100,100)
    model.load_state_dict(torch.load('lm-best.th'))
    hidden = model.init_hidden(1)
    
    pre_words = ['apple', 'i', 'today']

    for pre_word in pre_words:
        words = []
        print('first word:', pre_word)
        print('sentence created by NNLM:')
        words.append(pre_word)
        index = TEXT.vocab.stoi[pre_word]
        input = torch.LongTensor([[index]])
        for i in range(20):
            output, hidden = model(input, hidden)
            word_weights = output.squeeze().exp().cpu()
            word_idx = torch.multinomial(word_weights, 1)[0]
            input.fill_(word_idx)
            word = TEXT.vocab.itos[word_idx]
            words.append(word)
        print(" ".join(words))
        print()