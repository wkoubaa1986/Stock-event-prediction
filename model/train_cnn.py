

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pickle
import datetime
import pandas as pd
from cnn_config import *
import torch
import datetime
from ntn_pytorch import NeuralTensorNetwork_wrapper
from cnn_pytorch import LMS_CNN_wrapper
from prepocessing_cnn import DataGenerator_average_torch
import time

# start=time.time()
# with open('embeddedwords_average_20210912.pickle','rb') as f:
#     result=pickle.load(f)

# data=result['events']
# dictionary_size=len(result['events'])

# data=torch.from_numpy(np.double(data[:,:,:])).float()
# # train_set = torch.utils.data.TensorDataset(data)
# device = "cpu"
# cuda=False
# # kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
# # train_loader = torch.utils.data.DataLoader(
#  			dataset=train_set,
#  			batch_size=2*BATCH_SIZE, shuffle=False, **kwargs)

# tensor_dim_ntn=32
# # data=torch.load('./error/data_batch5')
# # data=data.cpu().detach().numpy()
# # data=torch.from_numpy(np.double(data)).float()
# # data=data.cpu().detach()
# # data_corru=torch.load('./error/data_corrupted_batch5')
# # data_corru=data_corru.cpu().detach().numpy()
# model = NeuralTensorNetwork_wrapper(dictionary_size,embedding_size_ntn,tensor_dim_ntn,dropout_ntn,LEARNING_RATE_NTN,device,cuda=cuda)
# model.load_weights('./tmp/ntn_trained_32_51_AverageLoss_0.99_Dataleft_0.0.model')
# data=model.predict(data)
# result['events']=data
# with open('embeddedwords_NTN_20210912.pickle', 'wb') as handle:
#     pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
# data_corru=model.predict(data_corru)
# ntn_news_embedding=model.predict(data)
# model.load_weights('./tmp/ntn_trained_32_8_AverageLoss_0.0_Dataleft_0.0.model')
# data=model.predict(data)
# data_corru=model.predict(data_corru)
# ntn_news_old=model.predict(data)
# ntn_news_old=ntn_news_old.cpu().detach().numpy()
# result={'comp_name':result['comp_events'],'dates_events':result['dates_events'],'label_events':result['label_events']}
# datagenerator=DataGenerator_average_torch(result,ntn_news_embedding)
# print('*****************Cnnn data processing finished**********************')
#



#*********************************CNN Training
with open('./input_cnn/cnn_input_SMALL.pickle','rb') as f:
    result=pickle.load(f)

data_base='SMALL'
concat=False
sparsity=False
cuda=False
if concat:
    EMBEDDING_DIM=result['long_term_concat'].shape[2]
else:
    EMBEDDING_DIM=result['long_term_mean'].shape[2]
BATCH_SIZE=100
datagenerator=DataGenerator_average_torch(result,concat=concat,sparsity=sparsity, onehot_target=False)
train_loader, test_loader = datagenerator.prepare_dataset_torch(cuda=cuda,batch_size=BATCH_SIZE)

model = LMS_CNN_wrapper()

model.fit(train_loader=train_loader,test_loader=test_loader,epochs=NUM_EPOCH)
model.save('./tmp/cnn_trained_database_'+data_base+'_concat_'+str(concat)+'_sparsity_'+str(concat)+'.model')


