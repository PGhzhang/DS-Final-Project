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

# construct user_friend trust matrix
n_users = 2099

user_matrix = np.zeros( (n_users,n_users), dtype=float) #np.zeros((n_users, n_items))
for row in user_friends.itertuples():
    user_matrix [row[1] - 2, row[2] - 2] = 1

# normalize user matrix
for row in range(user_matrix.shape[0]):
    count=sum(user_matrix[row])
    user_matrix[row] /= (count*1.0)

# calculating tag similarity
tag = pd.read_csv('data/tags.dat',sep='\t',engine='python')
timetag = pd.read_csv('data/user_taggedartists-timestamps.dat',sep='\t',engine='python')
timetag = timetag.merge(tag, left_on ='tagID', right_on = 'tagID', how = 'left')
timetag.head(50)
n_users = 2099
n_item = 18745

# a list of users and ids of their tagged songs
user_tag = [[[] for i in range(n_item)] for j in range(n_users)]
for i in range(timetag.shape[0]):
    user_tag[timetag.iloc[i,0]-2][timetag.iloc[i,1]-1].append(timetag.iloc[i,2])

# calcuate the amount of tags each user tagged
user_item_number = [0 for i in range(n_users)]
for i in range(len(user_tag)):
    for j in user_tag[i]:
        if len(j)!=0: user_item_number[i]+=1

# tag_sim for all users
tag_sim = [[0 for i in range(n_users)] for j in range(n_users)]
for i in range(len(user_tag)):
    print(i)
    tag_sim[i][i]=1
    for j in range(i+1,len(user_tag)):
        # compute tag_sim[i][j] & tag_sim[j][i]
        for k in range(len(user_tag[0])):
            if(len(user_tag[i][k])==0):continue
            set1 = set(user_tag[i][k])
            if(len(user_tag[j][k])==0):continue
            set2 = set(user_tag[j][k])
            intersect = set1 & set2
            if(len(intersect)!=0):tag_sim[i][j] += len(intersect)*len(intersect)/(len(set1)*len(set2))
        if(tag_sim[i][j]!=0): tag_sim[i][j] /= max(user_item_number[i],user_item_number[j])
        tag_sim[j][i] = tag_sim[i][j]

# filter tag_sim to only friends
tag_sim_copy = tag_sim.copy()
for i in range(user_matrix.shape[0]):
    for j in range(user_matrix.shape[1]):
        if user_matrix[i][j]==0:
            tag_sim_copy[i,j]=0
user_matrix_copy = lil_matrix(user_matrix.copy())



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

# calculate social influence
train_social = user_matrix.dot(train)

alpha = 0.4
for i in range(train.shape[0]):
    for j in range(train.shape[1]):
        if train[i][j] != 0:
            train[i][j] = alpha * train[i][j] + (1-alpha) * train_tag[i][j]

# calcualte social influence with tag_similarity amplication
tag_sim_copy = lil_matrix(tag_sim_copy)
train_tag_copy = tag_sim_copy.dot(train)

alpha = 0.2
beta = 0.4
for i in range(train.shape[0]):
    for j in range(train.shape[1]):
        if train[i][j] != 0:
            train[i][j] = alpha * train[i][j] + beta * train_social[i][j] + (1-alpha-beta)* train_tag_copy[i][j]

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

train = lil_matrix(train)
test = lil_matrix(test)

model,rmse = plainvanilla(train,test, EPOCH = 80, BATCH_SIZE = 100, LR = 0.1, l2_penalty=0.01, latent_factor=5)




