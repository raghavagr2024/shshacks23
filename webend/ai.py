import torch
import torch.nn as nn
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob

from transformers import BertTokenizer, BertModel

class BERTGRUSentiment(nn.Module):
    def __init__(self, bert, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super().__init__()

        self.bert = bert

        embedding_dim = bert.config.to_dict()['hidden_size']

        self.rnn = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=0 if n_layers < 2 else dropout,
        )

        self.out = nn.Linear(
            hidden_dim * 2 if bidirectional else hidden_dim, output_dim
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # text = [batch size, sent len]

        with torch.no_grad():
            embedded = self.bert(text)[0]

        # embedded = [batch size, sent len, emb dim]

        _, hidden = self.rnn(embedded)

        # hidden = [n layers * n directions, batch size, emb dim]

        if self.rnn.bidirectional:
            hidden = self.dropout(
                torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
            )
        else:
            hidden = self.dropout(hidden[-1, :, :])

        # hidden = [batch size, hid dim]

        output = self.out(hidden)

        # output = [batch size, out dim]

        # o = F.softmax(output, dim=1)

        return output
    
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe('spacytextblob')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert = BertModel.from_pretrained('bert-base-uncased')

HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.25

model = BERTGRUSentiment(
    bert, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT
)

model.load_state_dict( torch.load('model.pt', map_location='cpu') )
model.eval()

def get_my_rating(inp):
    tokens = tokenizer.tokenize(inp)

    if len(tokens) > 48:
        tokens = tokens[:48]
    else:
        tokens = tokens + [tokenizer.unk_token_id] * (48 - len(tokens))

    arr = torch.tensor(tokenizer.convert_tokens_to_ids( tokens ), dtype=torch.int32)

    output = model(arr)

    rating = torch.round(torch.sigmoid(output))

    return rating

def get_spacy_rating(inp):
    doc = nlp(inp)
    polarity = doc._.blob.polarity
    subjectivity = doc._.blob.subjectivity
    rating = subjectivity*0.5 + (polarity+1)*5
    return rating

def get_rating(inp):
    rating = (get_my_rating(inp) + get_spacy_rating(inp)) * 10/10.6
    return rating
