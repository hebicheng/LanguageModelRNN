import torch
import config
from RNNModel import RNNModel
from DataLoader import dataLoader
import numpy as np


def repackage_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def evaluate(model, data, VOCAB_SIZE, loss_fn):
    model.eval()
    total_loss = 0.
    it = iter(data)
    total_count = 0.
    with torch.no_grad():
        hidden = model.init_hidden(config.BATCH_SIZE, requires_grad=False)
        for i, batch in enumerate(it):
            data, target = batch.text, batch.target
            if config.USE_CUDA:
                data, target = data.cuda(), target.cuda()
            hidden = repackage_hidden(hidden)
            with torch.no_grad():
                output, hidden = model(data, hidden)
            loss = loss_fn(output.view(-1, VOCAB_SIZE), target.view(-1))
            total_count += np.multiply(*data.size())
            total_loss += loss.item()*np.multiply(*data.size())
            
    loss = total_loss / total_count
    model.train()
    return loss

def train():
    print("training...")
    VOCAB_SIZE, train_iter, test_iter, val_iter = dataLoader()
    model = RNNModel(vocab_size=VOCAB_SIZE, embed_size=config.EMBEDDING_SIZE, hidden_size=config.HIDDEN_SIZE)
    loss_fn = torch.nn.CrossEntropyLoss()
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)
    val_losses = []
    if config.USE_CUDA:
        model.to(torch.device('cuda'))

    for epoch in range(config.NUMBER_EPOCHS):
        model.train()
        hidden = model.init_hidden(config.BATCH_SIZE)
        it = iter(train_iter)
        for i, batch in enumerate(it):
            data, target = batch.text, batch.target
            if config.USE_CUDA:
                data, target = data.cuda(), target.cuda()
            hidden = repackage_hidden(hidden)
            model.zero_grad()
            output, hidden = model(data, hidden)
            # x = output.view(-1, VOCAB_SIZE)
            # print(x.shape)
            # exit()
            # pytorch document
            #   -This criterion combines nn.LogSoftmax() and nn.NLLLoss() in one single class.
            loss = loss_fn(output.view(-1, VOCAB_SIZE), target.view(-1))
            loss.backward()
            # 在循环神经网络中, 在梯度下降求导时, 带有指数项， 如果不做梯度裁剪，很容易发生梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            optimizer.step()
            if i % 100 == 0:
                print("epoch", epoch, "iter", i, "loss", loss.item())
        
            if i % 10000 == 0:
                val_loss = evaluate(model, val_iter,VOCAB_SIZE, loss_fn)
                
                if len(val_losses) == 0 or val_loss < min(val_losses):
                    print("model saved, val loss: ", val_loss)
                    torch.save(model.state_dict(), "lm-best.th")
                else:
                    scheduler.step()
                    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                val_losses.append(val_loss)