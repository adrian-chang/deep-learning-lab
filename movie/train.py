import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import logging
logging.basicConfig(level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# force the seed to be the same
np.random.seed(0)

def train_test_datasets(train_size=0.7):
    logger.info('Loading reviews')
    with open('./binary-reviews.npz', 'rb') as file:
        reviews = np.load(file)['arr_0']
    logger.info('Loading classifications')
    with open('./binary-labels.npz', 'rb') as file:
        review_classifications = np.load(file)['arr_0']
    ### CRITICAL https://gist.githubusercontent.com/Tushar-N/dfca335e370a2bc3bc79876e6270099e/raw/940e8a4146f6424f842e25d034c2c1ac3440f84d/pad_packed_demo.py
    ### Must use padded sequence for variable sizes
    record_count, record_features = reviews.shape[0], reviews.shape[1]
    logger.info(f'Using {record_count} reviews of {record_features} words')
    split = np.random.permutation(record_count)
    amount = review_classifications.shape[0]
    test_id = split[int(amount * train_size):]
    train_id = split[:int(amount * train_size)]
    train = reviews[train_id, :]
    train_classify = review_classifications[train_id]
    test = reviews[test_id, :]
    test_classify = review_classifications[test_id]
    return train, train_classify, test, test_classify

class MovieDataset(Dataset):

    def __init__(self, data, label):
        super(MovieDataset, self).__init__()
        self._data = data
        self._label = label

    def __getitem__(self, index):
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
        # careful of leak index
        lengths = []
        for i, review in enumerate(reviews):
            lengths.append((i, (review == 0).nonzero().data[0]))
        #lengths = t.LongTensor(lengths).cuda()
        reordered_reviews = t.zeros(reviews.shape).long().cuda()
        lengths = sorted(lengths, key=lambda item: item[1], reverse=True)
        non_zero_lengths = []
        for i, (org_index, length) in enumerate(lengths):
            reordered_reviews[i] = reviews[org_index]
            non_zero_lengths.append(length)
        non_zero_lengths = t.LongTensor(non_zero_lengths).cuda()
        embedded_reviews = self._embedding(reordered_reviews).transpose(0, 1) # watch out here, it's b, s, l, needs to be s, b, l
        embedded_reviews = pack_padded_sequence(embedded_reviews, non_zero_lengths)
        # output [-1] === hidden last
        output, (ht, ct) = self._lstm(embedded_reviews, self._hidden) # use output, needs to pad_pack
        #self._hidden = (hidden[0].data.clone(), hidden[1].data.clone())
        predictions = self._linear(ht[-1])
        return predictions

def train_model(model, dataloader, epochs = 4):
    logger.info('Training model')
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    for epoch in tqdm(range(epochs)):
        for i, (movie_reviews, movie_labels) in enumerate(dataloader):
            model.zero_grad() # DO NOT DO OPT ZERO_GRAD
            predictions = model(movie_reviews)
            loss = criterion(predictions, movie_labels.long().cuda())
            #print(predictions, movie_labels, loss)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                logger.info(f'{i} {loss.item()}')

def test_model(model, dataloader):
    logger.info('Testing Model')
    model.eval()
    correct = 0
    total = 0
    for movie_reviews, movie_labels in tqdm(dataloader):
        movie_reviews = movie_reviews.long().cuda()
        movie_labels = movie_labels.long().cuda()
        preds = model.forward(movie_reviews)
        val, idx = preds.max(1)
        total += idx.size()[0]
        correct += idx.eq(movie_labels.view_as(idx)).cpu().sum()
    logger.info(f'correct {correct.cpu().numpy() / total}')

if __name__ == '__main__':
    logger.info('Running program')
    train_x, train_y, test_x, test_y = train_test_datasets()
    WORD_TOTAL = 45636 + 1
    #WORD_TOTAL = 50000
    REVIEW_LENGTH = 500
    BATCH_SIZE = 500 ## See above note for critical for variable sizes
    movie_dataset = MovieDataset(train_x[:1000], train_y[:1000])
    movie_dataloader = DataLoader(movie_dataset, batch_size=BATCH_SIZE)
    model = Sentiment(REVIEW_LENGTH, WORD_TOTAL, BATCH_SIZE).cuda()
    train_model(model, movie_dataloader)
    t.save(model.state_dict(), './model-final.pt')
    model.load_state_dict(t.load('./model-final.pt'))
    test_model(model, movie_dataloader)
