import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

import torchtext
from torchtext import data

from tqdm.auto import tqdm
import pandas as pd

#from torchtext.transforms import BERTTokenizer
from transformers import BertTokenizer, BertModel
#from torchtext.vocab import build_vocab_from_iterator

#VOCAB = "https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt"


class BERTGRUSentiment(nn.Module):
    def __init__(self,
                 bert,
                 hidden_dim,
                 output_dim,
                 n_layers,
                 bidirectional,
                 dropout):
        
        super().__init__()
        
        self.bert = bert
        
        embedding_dim = bert.config.to_dict()['hidden_size']
        
        self.rnn = nn.GRU(embedding_dim,
                          hidden_dim,
                          num_layers = n_layers,
                          bidirectional = bidirectional,
                          batch_first = True,
                          dropout = 0 if n_layers < 2 else dropout)
        
        self.out = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        
        #text = [batch size, sent len]
                
        with torch.no_grad():
            embedded = self.bert(text)[0]
                
        #embedded = [batch size, sent len, emb dim]
        
        _, hidden = self.rnn(embedded)
        
        #hidden = [n layers * n directions, batch size, emb dim]
        
        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
                
        #hidden = [batch size, hid dim]
        
        output = self.out(hidden)
        
        #output = [batch size, out dim]
        
        return output


class SentimentDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.df = pd.read_csv('./data/sentiment140_shrunk.csv')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.init_token_idx = self.tokenizer.cls_token_id
        self.eos_token_idx = self.tokenizer.sep_token_id
        self.pad_token_idx = self.tokenizer.pad_token_id
        self.unk_token_idx = self.tokenizer.unk_token_id

        self.text = self.df['text']
        self.polarity = self.df['polarity']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        try:
            tokens = self.tokenizer.tokenize(self.text[index])
            if len(tokens) > 510:
                tokens = tokens[:510]
            else:
                tokens = tokens + [self.unk_token_idx] * (510 - len(tokens))
            return torch.tensor(self.tokenizer.convert_tokens_to_ids( tokens ), dtype=torch.int32), torch.tensor(self.polarity[index], dtype=torch.float32)
        except:
            raise StopIteration()

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum() / len(correct)
    return acc

def t(model, dl, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for i, batch in enumerate(tqdm(dl)):
        text, polarity = batch
        text = text.cuda()
        polarity = polarity.cuda()

        optimizer.zero_grad()

        predictions = model(text).squeeze(1)
        
        loss = criterion(predictions, polarity)
        
        acc = binary_accuracy(predictions, polarity)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        if i % 50 == 0:
            tqdm.write(f'Loss: {epoch_loss / (i+1)} - Acc: {epoch_acc / (i+1)}')
        if i % 1000 == 0:
            torch.save(model.state_dict(), f'saves/model_at_{i}.pt')
    return epoch_loss / len(dl), epoch_acc / len(dl)


if __name__ == '__main__':
    bert = BertModel.from_pretrained('bert-base-uncased')

    HIDDEN_DIM = 256
    OUTPUT_DIM = 1
    N_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.25

    model = BERTGRUSentiment(
        bert,
        HIDDEN_DIM,
        OUTPUT_DIM,
        N_LAYERS,
        BIDIRECTIONAL,
        DROPOUT
    )
    for name, param in model.named_parameters():                
        if name.startswith('bert'):
            param.requires_grad = False
    
    print(count_params(model))

    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.BCEWithLogitsLoss()

    model.cuda()
    criterion.cuda()

    ds = SentimentDataset()
    dl = DataLoader(ds, batch_size=96, num_workers=16, shuffle=True, prefetch_factor=32, pin_memory=True)


    print(t(model, dl, optimizer, criterion))

    torch.save(model.state_dict(), 'model.pt')
    torch.save(optimizer.state_dict(), 'optim.pt')
