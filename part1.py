import numpy as np
import math
import argparse
from scipy.sparse import rand as sprand
from scipy.sparse import lil_matrix
import torch
from torch.autograd import Variable
import pandas as pd

user_friends = pd.read_csv('data/user_friends.dat',sep='\t',engine='python')
user_artists = pd.read_csv('data/user_artists.dat',sep='\t',engine='python')

user_artists.head()
# create a table with total number of plays
user_plays = (user_artists.
              groupby(by = ['userID'])['weight'].
              sum().
              reset_index().
              rename(columns = {'weight': 'total_user_plays'})
              [['userID', 'total_user_plays']])
user_plays.head()

user_artists_with_total_plays = user_artists.merge(user_plays, left_on ='userID', right_on = 'userID', how = 'left')
user_artists_with_total_plays

user_artists_with_total_plays['prerank'] = user_artists_with_total_plays['weight']/user_artists_with_total_plays['total_user_plays']

user_artists_with_total_plays['rank'] = user_artists_with_total_plays.groupby('userID')['prerank'].rank(ascending=True)

df = user_artists_with_total_plays

rating=[]
for i in range(user_artists_with_total_plays.shape[0]):
    if i==0:
        rating.append(5)
        temp=df.iloc[i][4]
    else:
        if df.iloc[i][0]==df.iloc[i-1][0]:
            rating.append(5*(1-temp))
            temp+=df.iloc[i][4]
        else:
            rating.append(5)
            temp=df.iloc[i][4]


df['rating']=rating

from sklearn.model_selection import train_test_split
rawtrain, rawtest = train_test_split(df, test_size=0.3)

def get_artist_ratings(df):
    #     n_users = max(df.userID.unique())
    #     n_items = max(df.artistID.unique())
    n_users = 2099
    n_items = 18745
    
    interactions = np.zeros( (n_users,n_items), dtype=float) #np.zeros((n_users, n_items))
    for row in df.itertuples():
        interactions[row[1] - 2, row[2] - 2] = row[7]
    return interactions

train = get_artist_ratings(rawtrain)
test = get_artist_ratings(rawtest)


# Collaboritive Filtering

from sklearn.metrics.pairwise import pairwise_distances
user_similarity = pairwise_distances(train, metric='cosine')
item_similarity = pairwise_distances(train.T, metric='cosine')

def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        ratings_diff = (ratings - mean_user_rating)
        pred = mean_user_rating + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred

user_prediction = predict(train, user_similarity, type='user')

from sklearn.metrics import mean_squared_error
from math import sqrt
def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].toarray()
    return sqrt(mean_squared_error(prediction, ground_truth))

print ('User-based CF RMSE: ' + str(rmse(user_prediction, test)))


# Matrix Factorization

class MatrixFactorization(torch.nn.Module):
    
    def __init__(self, n_users, n_items, n_factors=5):
        super().__init__()
        self.user_factors = torch.nn.Embedding(n_users,
                                               n_factors,
                                               sparse=False)
                                               self.item_factors = torch.nn.Embedding(n_items,
                                                                                      n_factors,
                                                                                      sparse=False)

# For convenience when we want to predict a sinble user-item pair.
def predict(self, users, items):
    pred = torch.mm(users,self.item_factors(items))
        pred = torch.mm(pred,torch.transpose(self.item_factors(items),0,1))
        return pred
    
    # Much more efficient batch operator. This should be used for training purposes
    def forward(self, users, items):
        # Need to fit bias factors
        return torch.mm(self.user_factors(users),torch.transpose(self.item_factors(items),0,1))

def get_batch(batch_size,ratings):
    # Sort our data and scramble it
    rows, cols = ratings.shape
    p = np.random.permutation(rows)
    
    # create batches
    sindex = 0
    eindex = batch_size
    while eindex < rows:
        batch = p[sindex:eindex]
        temp = eindex
        eindex = eindex + batch_size
        sindex = temp
        yield batch
    
    if eindex >= rows:
        batch = range(sindex,rows)
        yield batch


def test_error(model,test,BATCH_SIZE):
    loss_func = torch.nn.MSELoss()
    square_deviation = 0
    msei=0
    for i,batch in enumerate(get_batch(BATCH_SIZE, test)):
        # Turn data into variables
        interactions = Variable(torch.FloatTensor(test[batch, :].toarray()))
        rows = Variable(torch.LongTensor(batch))
        cols = Variable(torch.LongTensor(np.arange(test.shape[1])))
        
        # Predict and calculate loss
        predictions = model.predict(interactions, cols)
        loss = loss_func(predictions, interactions)
        
        # plus the square deviation
        square_deviation += loss*rows.shape[0]*cols.shape[0]
        msei += rows.shape[0]*cols.shape[0]
    rmse = torch.sqrt(square_deviation/msei)
    print("Test RMSE loss", rmse)
    return rmse

def plainvanilla(train, test, EPOCH = 100, BATCH_SIZE = 1000, LR = 0.1,l2_penalty=0.01,latent_factor=3):
    model = MatrixFactorization(train.shape[0], train.shape[1], n_factors=latent_factor)
    loss_func = torch.nn.MSELoss()
    reg_loss_func = torch.optim.SGD(model.parameters(), lr=LR, weight_decay = l2_penalty)
    for i in range(EPOCH):
        print("Epoch:", i)
        square_deviation = 0
        msei=0
        for j,batch in enumerate(get_batch(BATCH_SIZE, train)):
            # Set gradients to zero
            reg_loss_func.zero_grad()
            
            # Turn data into variables
            
            interactions = Variable(torch.FloatTensor(train[batch, :].toarray()))
            rows = Variable(torch.LongTensor(batch))
            cols = Variable(torch.LongTensor(np.arange(train.shape[1])))
            
            # Predict and calculate loss
            predictions = model(rows, cols)
            loss = loss_func(predictions, interactions)
            
            # Backpropagate
            loss.backward()
            
            # Update the parameters
            reg_loss_func.step()
            
            # plus the square deviation
            if i==EPOCH-1:
                square_deviation += loss*rows.shape[0]*cols.shape[0]
                msei += rows.shape[0]*cols.shape[0]
        print(loss)
    
    test_rmse = test_error(model,test,BATCH_SIZE)
    return model,test_rmse

model,rmse = plainvanilla(train,test, EPOCH = 80, BATCH_SIZE = 100, LR = 0.1, l2_penalty=0.01, latent_factor=5)




