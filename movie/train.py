import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

# force the seed to be the same
np.random.seed(0)

def train_test_datasets(train_size=0.7):
    print('Loading classifications as review_classifications')
    with open('../../text-classification-in-pytorch-using-lstm/tc-class.npz', 'rb') as file:
        review_classifications = np.load(file)['arr_0']
    print('Loading reviews as reviews_as_indexes')
    with open('../../text-classification-in-pytorch-using-lstm/tc-reviews.npz', 'rb') as file:
        reviews_as_indexes = np.load(file)['arr_0']
    ### CRITICAL https://gist.githubusercontent.com/Tushar-N/dfca335e370a2bc3bc79876e6270099e/raw/940e8a4146f6424f842e25d034c2c1ac3440f84d/pad_packed_demo.py
    ### Must use padded sequence for variable sizes
    '''
    stored = []
    for row in reviews_as_indexes:
        local = np.array(row)
        local = local + 1
        local.resize(500)
        stored.append(local)

    stored = np.array(stored)
    with open('./tc-reviews.npz', 'wb') as file:
        np.savez(file, stored)
    '''

    return reviews_as_indexes, review_classifications, reviews_as_indexes, review_classifications
    #split = np.random.permutation(review_classifications.shape[0])
    #amount = review_classifications.shape[0]
    #test_id = split[int(amount * train_size):]
    #train_id = split[:int(amount * train_size)]
    #train = reviews_as_indexes[train_id, :]
    #train_classify = review_classifications[train_id]
    #test = reviews_as_indexes[test_id, :]
    #test_classify = review_classifications[test_id]
    #return train, train_classify, test, test_classify

class MovieDataset(Dataset):

    def __init__(self, data, label):
        super(MovieDataset, self).__init__()
        self._data = data
        self._label = label

    def __getitem__(self, index):
        print(self._data[index])
        return self._data[index], self._label[index]

    def __len__(self):
        return self._data.shape[0]

class Sentiment(nn.Module):

    def __init__(self, review_length, word_total, batch_size=5, embedding_dim=300, hidden_dim=100, dropout=0.2, classifications = 2):
        super(Sentiment, self).__init__()
        self._hidden_dim = hidden_dim
        self._batch_size = batch_size
        self._classifications = classifications
        self._dropout = 0.2 # unused right now
        # https://discuss.pytorch.org/t/illegal-memory-access-in-backward-after-first-training-epoch/920
        # we need to use nn.DataParallel ? on lSTM? thought cuda would do it, but there's a bug
        # bug happens when embedding word_total is not accurate (overflow)
        self._embedding = nn.Embedding(word_total, embedding_dim)
        self._lstm = nn.LSTM(embedding_dim, hidden_dim)
        self._linear = nn.Linear(hidden_dim, classifications)
        # layers = reviews because of embedding, rows = words, cols = emb
        self._hidden = (t.zeros(1, self._batch_size, self._hidden_dim).cuda(), t.zeros(1, self._batch_size, self._hidden_dim).cuda())

    def forward(self, reviews):
        embedded_reviews = self._embedding(reviews)
        # output [-1] === hidden last
        output, hidden = self._lstm(embedded_reviews.t(), self._hidden)
        #self._hidden = (hidden[0].data.clone(), hidden[1].data.clone())
        predictions = self._linear(output[-1])
        return predictions

def train_model(model, dataloader, epochs = 1):
    print('Training model')
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    for epoch in tqdm(range(epochs)):
        for i, (movie_reviews, movie_labels) in enumerate(dataloader):
            #print(movie_reviews, movie_labels)
            predictions = model(t.Tensor([movie_reviews]).long().cuda())
            model.zero_grad() # DO NOT DO OPT ZERO_GRAD
            loss = criterion(predictions, t.Tensor([movie_labels]).long().cuda())
            #print(predictions, movie_labels, loss)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f'{i} {loss.item()}')n

def test_model(model, dataloader):
    print('Testing Model')
    model.eval()
    correct = 0
    total = 0
    for movie_review, movie_label in tqdm(dataloader):
        movie_review = t.Tensor([movie_review]).long().cuda()
        movie_label = t.Tensor([movie_label]).long().cuda()
        preds = model.forward(movie_review)
        val, idx = preds.max(1)
        total += idx.size()[0]
        correct += idx.eq(movie_label.view_as(idx)).cpu().sum()
    print(f'correct {correct.cpu().numpy() / total}')

if __name__ == '__main__':
    print('Running program')
    train_x, train_y, test_x, test_y = train_test_datasets()
    #WORD_TOTAL = 45636 + 1
    WORD_TOTAL = 50000
    REVIEW_LENGTH = 5000
    BATCH_SIZE = 1 ## See above note for critical for variable sizes
    movie_dataset = MovieDataset(train_x[:10], train_y[:10])
    movie_dataloader = DataLoader(movie_dataset, batch_size=BATCH_SIZE)
    model = Sentiment(REVIEW_LENGTH, WORD_TOTAL, BATCH_SIZE).cuda()
    train_model(model, movie_dataloader)
    t.save(model.state_dict(), './model-3.pt')
    #model.load_state_dict(t.load('./model-2.pt'))
    test_model(model, movie_dataloader)
