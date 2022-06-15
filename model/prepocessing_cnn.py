from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pickle
import datetime
from cnn_config import LONG_TERM_LENGTH, TRAIN_TEST_SEP_RATE, MID_TERM_LENGTH, \
	EMBEDDING_DIM
import torch
import datetime
import time

class EventEmbedding_average(object):

    def __init__(self, result,concat,sparsity, onehot_target=False):

        self.one_hot_target = onehot_target
        
        #self.ntn_news_embedding = []
        self.dataframe = result['dataframe']
        if concat:
            self.long_term=result['long_term_concat']
        else:
            self.long_term=result['long_term_mean']
        self.sparsity=sparsity  
        self.label_data = []
            

        self.get_data()

    
        
    def get_data(self):
        def one_hot_encode(int_label):
            assert int_label in (0, 1)
            return [[0, 1], [1, 0]][int_label]
      
        if self.sparsity:
            ind=np.lexsort((self.dataframe[:,1],self.dataframe[:,0],self.dataframe[:,4],self.dataframe[:,3]))
            ind=np.squeeze(np.rot90(np.expand_dims(ind,axis=1),k=2),1)
            self.long_term = np.array(self.long_term[ind,:,:])
            self.label_data=np.array(self.dataframe[ind,2])
            self.dataframe= np.transpose(np.asarray([self.dataframe[ind,0],self.dataframe[ind,1],self.dataframe[ind,2],self.dataframe[ind,3],self.dataframe[ind,4]]))
            #.... Reducing to the less sparsed data:
            N=50000
            self.long_term=self.long_term[0:N,:,:]
            self.dataframe=self.dataframe[0:N,:]
            self.label_data=self.label_data[0:N]
        else:
            self.label_data=np.array(self.dataframe[:,2])
    
        if self.one_hot_target:
            self.label_data = np.array([one_hot_encode(int(max(0,x))) for x in self.label_data])  # TODO remove sklearn normalize here
        else:
            self.label_data = np.array([max(0.,float(x)) for x in self.label_data])
             
            
      
        print('Getting Long term data finished input finished')
    def prepare_dataset(self):
        
        separation_rate = TRAIN_TEST_SEP_RATE
        t = int(len(self.label_data) * separation_rate)
        label_train = self.label_data[:t]
        label_test = self.label_data[t:]

		# train data,  each item in the input_train has a full spectrum of 30 days news embedding
        long_term_train = np.array(self.long_term[:t,:,:])  # long term
        label_train_array = np.array(label_train) 
        train_data = {"long_term": long_term_train}# we don't need the other it will be reconstrcted
        long_term_test = np.array(self.long_term[t:,:,:])  # long term
        label_test_array = np.array(label_test) 

        test_data = {"long_term": long_term_test}  # , label_test)
        print('-Whole set label long: {} % short: {} %'.format(float(round(len(np.where(self.label_data==1.)[0])/len(self.label_data)*100,2)),float(round(len(np.where(self.label_data==0.)[0])/len(self.label_data)*100,2))))
        print('-Train set label long: {} % short: {} %'.format(float(round(len(np.where(label_train==1.)[0])/len(label_train)*100,2)),float(round(len(np.where(label_train==0.)[0])/len(label_train)*100,2))))
        print('-Test set label long: {} % short: {} %'.format(float(round(len(np.where(label_test==1.)[0])/len(label_test)*100,2)),float(round(len(np.where(label_test==0.)[0])/len(label_test)*100,2))))

        return (train_data, label_train_array), (test_data, label_test_array)


class DataGenerator_average_torch(EventEmbedding_average):
    def __init__(self,  result,concat,sparsity, onehot_target=False):
        super(DataGenerator_average_torch, self).__init__(result=result,concat=concat,sparsity=sparsity, onehot_target=False)
    
    def extract_convert_to_torch(self,data,targets):
        long_term = data["long_term"]
        long_term=long_term.reshape((long_term.shape[0],1,long_term.shape[1],long_term.shape[2]))
        long_term=torch.from_numpy(np.double(long_term)).float()
        out_targets = torch.squeeze(torch.from_numpy(targets)).long()
        return (long_term,out_targets ) 
    
    def prepare_dataset_torch(self, cuda, batch_size):
        (train_data, train_targets), (test_data, test_targets) = self.prepare_dataset()
        long_term_train,targets_train  =self.extract_convert_to_torch(train_data,train_targets)
        long_term_test,targets_test  =self.extract_convert_to_torch(test_data,test_targets)
        #x_train = x_train.reshape((x_train.shape[0], 1, LONG_TERM_LENGTH, EMBEDDING_DIM * MAX_SEQUENCE_LENGTH))


        kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}

        train_set = torch.utils.data.TensorDataset(long_term_train, targets_train)
        test_set = torch.utils.data.TensorDataset(long_term_test, targets_test)
        
        train_loader = torch.utils.data.DataLoader(
			dataset=train_set,
			batch_size=batch_size, shuffle=True, **kwargs)
        
        test_loader = torch.utils.data.DataLoader(
			dataset=test_set,
			batch_size=batch_size, shuffle=True, **kwargs)

        return train_loader, test_loader

    
    
if __name__ == "__main__":
    
    with open('./input_cnn/cnn_input_SMALL.pickle','rb') as f:
        result=pickle.load(f)
    
    
    #******************************************************************
    EMBEDDING_DIM=100
    BATCH_SIZE=250
    datagenerator=DataGenerator_average_torch(result,concat=True,sparsity=True, onehot_target=False)
    train_loader, test_loader = datagenerator.prepare_dataset_torch(cuda=True,batch_size=BATCH_SIZE)
    


