import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
#from torchvision import datasets, transforms
from torch.autograd import Variable
import random
from tqdm import tqdm
import pickle
import time
import keyboard

class NeuralTensorNetwork(nn.Module):

    def __init__(self, dictionary_size, embedding_size, tensor_dim, dropout, device="cpu",cuda=False):
        super(NeuralTensorNetwork, self).__init__()
        

        self.device = device
        #self.emb = nn.Embedding(dictionary_size, embedding_size)
        self.tensor_dim = tensor_dim
        self.embedding_size=embedding_size
        ##Tensor Weight
        # |T1| = (embedding_size, embedding_size, tensor_dim)
        self.T1 = nn.Parameter(torch.Tensor(embedding_size * embedding_size * tensor_dim))
        #self.T1 = torch.reshape(self.T1,(embedding_size,embedding_size,tensor_dim))
        self.T1.data.normal_(mean=0.0, std=0.02)

        # |T2| = (embedding_size, embedding_size, tensor_dim)
        self.T2 = nn.Parameter(torch.Tensor(embedding_size * embedding_size * tensor_dim))
        #self.T2 = torch.reshape(self.T2,(embedding_size,embedding_size,tensor_dim))
        self.T2.data.normal_(mean=0.0, std=0.02)

        # |T3| = (tensor_dim, tensor_dim, tensor_dim)
        self.T3 = nn.Parameter(torch.Tensor(tensor_dim * tensor_dim * tensor_dim))
        #self.T3 = torch.reshape(self.T3,(tensor_dim,tensor_dim,tensor_dim))
        self.T3.data.normal_(mean=0.0, std=0.02)

        # |W1| = (embedding_size * 2, tensor_dim)
        self.W1 = nn.Linear(embedding_size * 2, tensor_dim)

        # |W2| = (embedding_size * 2, tensor_dim)
        self.W2 = nn.Linear(embedding_size * 2, tensor_dim)

        # |W3| = (tensor_dim * 2, tensor_dim)
        self.W3 = nn.Linear(tensor_dim * 2, tensor_dim)

        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=dropout)
        # attention mechanism
        #self.linear=nn.Linear(tensor_dim,1)
        

    def forward(self, data):
        # |svo| = (batch_size, max_length)
        # |sov_length| = (batch_size, 3)

        #svo = self.emb(data)
        # |svo| = (batch_size, max_lenght, embedding_size)

        ## To merge word embeddings, Get mean value
        subj = data.narrow(1,0,1)
        verb = data.narrow(1,1,1)
        obj = data.narrow(1,2,1)
        
        subj.contiguous()
        verb.contiguous() 
        obj.contiguous()

        subj=subj.view(subj.size(0), -1)
        verb=verb.view(verb.size(0), -1)
        obj=obj.view(obj.size(0), -1)
        # |subj|, |verb|, |obj| = (batch_size, embedding_size)

        R1 = self.tensor_Linear(subj, verb, self.T1, self.W1)
        R1 = self.tanh(R1)
        R1 = self.dropout(R1)
        # |R1| = (batch_size, tensor_dim)

        R2 = self.tensor_Linear(verb, obj, self.T2, self.W2)
        R2 = self.tanh(R2)
        R2 = self.dropout(R2)
        # |R2| = (batch_size, tensor_dim)

        U = self.tensor_Linear(R1, R2, self.T3, self.W3)
        U = self.tanh(U)


        return U

    def tensor_multiplication(self,o1,o2,tensor_layer):
        
        for k in range(self.tensor_dim):
            o1Tk=torch.mm(o1,tensor_layer[:,:,k])
            o1Tk=o1Tk.unsqueeze(dim=1)
            o2k=o2.unsqueeze(dim=2)
            
            if k==0:
                tensor=torch.bmm(o1Tk,o2k).squeeze(2)
            else:
                tensor=torch.cat((tensor,torch.bmm(o1Tk,o2k).squeeze(2)),dim=1)
        
        return tensor
    
    def tensor_Linear(self, o1, o2, tensor_layer, linear_layer):
        # |o1| = (batch_size, unknown_dim)
        # |o2| = (batch_size, unknown_dim)
        # |tensor_layer| = (unknown_dim * unknown_dim * tensor_dim)
        # |linear_layer| = (unknown_dim * 2, tensor_dim)
       
        
        batch_size, embedding_size = o1.size()

        # 1. Linear Production
        o1_o2 = torch.cat((o1, o2), dim=1)
        # |o1_o2| = (batch_size, unknown_dim * 2)
        linear_product = linear_layer(o1_o2)
        
        #o1Tk
        tensor_product = o1.mm(tensor_layer.view(embedding_size, -1))
        tensor_product = tensor_product.view(batch_size, -1, embedding_size).bmm(o2.unsqueeze(1).permute(0,2,1).contiguous()).squeeze()

        
        # |linear_product| = (batch_size, tensor_dim)
        tensor_product = tensor_product.contiguous()
        # 3. Summation
        result = tensor_product + linear_product
        # |result| = (batch_size, tensor_dim)

        return result
    
    
class NeuralTensorNetwork_wrapper:
    def __init__(self,dictionary_size,embedding_size,tensor_dim,dropout,LEARNING_RATE,device='cpu',cuda=False):
        self.datasize=dictionary_size
        self.cuda = cuda
        self._model = NeuralTensorNetwork(dictionary_size,embedding_size,tensor_dim,dropout,device,cuda)

        self.optimizer = optim.Adam(self._model.parameters(), lr=LEARNING_RATE, )
        if self.cuda:
            self._model.cuda()
    def model_loss(self,E,E_corr,REGULARIZER_WEIGHT):
        l2_reg = None
        for W in self._model.parameters():
            if l2_reg is None:
                l2_reg = W.norm(2)
            else:
                l2_reg = l2_reg + W.norm(2)
        loss = l2_reg * REGULARIZER_WEIGHT
        #E=torch.squeeze(E)
        #E_corr=torch.squeeze(E_corr)
        T0=torch.zeros(E.shape)
        T1=torch.ones(E.shape)
        if self.cuda:
            T0=T0.cuda()
            T1=T1.cuda()
        loss_E_Er=torch.mean(torch.max(T0,torch.add(torch.sub(T1,E),E_corr)),1)
        loss_T=loss+torch.sum(loss_E_Er)
        #loss_T=loss+loss_E_Er
        return loss_E_Er, loss_T

    def train(self, train_loader,REGULARIZER_WEIGHT,dictionary_size):
        self._model.train()  # sets to train mode
        loss_list = 0
        iter_num = 0
        left=dictionary_size
        start=time.time()
        num_itr_loss=0
        for batch_idx, (data, data_corr) in enumerate(train_loader):
            #N_sample=train_loader[:,0,:].shape[0]     
            iter_batch=0
            o1_corr_re=data_corr[:,0,:].clone()
            diff_torsor=o1_corr_re-data[:,0,:].clone()
            #batch_size=len
            #verify if the tensor do not include any same input
            for i in range(0,len(o1_corr_re)):
                while torch.equal(diff_torsor[i,:],torch.zeros(diff_torsor[i,:].shape)):
                    o1_corr_re[i,:]=data_corr[torch.randint(len(data_corr),(1,)),0,:]
                    diff_torsor=o1_corr_re-data[:,0,:]
            data_corr[:,0,:]=o1_corr_re  
            exist_loop=False
            len_indx=len(data)
            #Max_inter = len(data) * 100
            
            while len_indx> 0 and not(exist_loop):
                iter_num += 1
                iter_batch+=1
                batch_size=data.shape[0]
                #data=train_loader[index,:,:]
                #data_corr_i=data_corr[index,:,:]
                if self.cuda:
                    data= data.cuda()
                    data_corr=data_corr.cuda()
                    
                
                try:
                    E=self._model(data)
                    E_corr=self._model(data_corr)
                except:
                    print('error')
                loss_E_Er, loss=self.model_loss(E, E_corr, REGULARIZER_WEIGHT)
                
                indx=loss_E_Er.nonzero()
                len_indx=len(indx)
                indx=indx.squeeze()

                left+=len_indx-len(data)
                num_itr_loss+=len_indx+1
                #if self.cuda:
                loss_list=((iter_num-1)*loss_list+loss.cpu().detach().numpy()/batch_size)/iter_num
            
                if len_indx==0:
                    if batch_idx  % 20 == 0:
                        print("batch number %d finished there are %f %% left in this epoch at iteration %d with average cost %f" % (batch_idx,float(left / dictionary_size) * 100, iter_num, loss_list))
                        end = time.time()
                        print('Time:'+str(round(end - start,2)) +' seconds')

                
                else:
                    try:
                            data=data[indx][:][:]
                            data_corr=data_corr[indx][:][:]
                            #loss_E_Er=loss_E_Er[indx][:][:]
                            if len(data.shape)==2:
                                data=data.unsqueeze(dim=0)
                                data_corr=data_corr.unsqueeze(dim=0)
                                T0=torch.zeros(data[:,0,:].shape)
                                if self.cuda:
                                    T0=T0.cuda()
                                if torch.equal(data[:,0,:]-data_corr[:,0,:],T0):
                                    data_corr[:,0,:].data.normal_(mean=0.0, std=0.04)
                                    
                    except:
                       print('error')

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    if iter_num % 5000 == 0:
                        print("There are %f %% left in this epoch at iteration %d with average cost %f" % (float(left / dictionary_size) * 100, iter_num, loss_list))
                        end = time.time()
                        print('Time:'+str(round(end - start,2)) +' seconds')
                    
                    if iter_batch>40000:
                        print('Did not converge')
                        torch.save(data,'./error/data_batch'+str(batch_idx))
                        torch.save(data,'./error/data_corrupted_batch'+str(batch_idx))
                        self.save('./error/modelweight_batch'+str(batch_idx))
                        #keyboard.wait('esc')
                        exist_loop=True

    
        return (np.mean(loss_list), float(left / dictionary_size) *100)
	
    def predict(self, test_loader):
        self._model.eval()  # sets model to test mode
        #output_all=None
        #for batch_idx, data in enumerate(test_loader):
        if self.cuda:
            data = test_loader.cuda()
        data = Variable(test_loader)
        with torch.no_grad():
            
            
            output = self._model(data)
            #if output_all is None:
              #  output_all=output
            #else:
              #  output_all=torch.stack([output_all,output],dim=0)
            
            return output.cpu().detach().numpy()
            #pred = output.data.max(1, )[1]

    def fit(self, data, epochs,REGULARIZER_WEIGHT,dictionary_size,batch_size):
        pbar = tqdm(range(1, epochs + 1))
        n=2
        
        # rechuffling for each epoch
        data_corr=data.copy()
        N_sample=data[:,0,:].shape[0]
        Vocabulary=np.concatenate((data_corr[:,0,:], data_corr[:,1,:], data_corr[:,2,:]),axis=0)
        N_Vocab=Vocabulary.shape[0]
        set_index = list(range(N_Vocab))
        data_corr[:,0,:]=Vocabulary[random.choices(set_index,k=N_sample),:]
        del Vocabulary, N_Vocab, set_index
        data_corr=torch.from_numpy(np.double(data_corr)).float()
        data=torch.from_numpy(np.double(data)).float()
        kwargs = {'num_workers': 4, 'pin_memory': True} if self.cuda else {}
        train_set = torch.utils.data.TensorDataset(data, data_corr)
        #train_set = torch.utils.data.TensorDataset(data)
        train_loader = torch.utils.data.DataLoader(
			dataset=train_set,
			batch_size=batch_size, shuffle=True, **kwargs)
        losses=list()
        for epoch in pbar:
            n+=1
            train_info = "Training set: Average loss:"
            average_loss, data_left = self.train(train_loader,REGULARIZER_WEIGHT,dictionary_size)
            #average_loss=average_loss/dictionary_size
            train_info += str(average_loss)
            self.save('./tmp/ntn_trained_32_'+str(n)+'_AverageLoss_'+str(round(np.mean(average_loss),2))+'_Dataleft_'+str(round(data_left,2))+'.model')
            losses.append(average_loss)
            with open('./tmp/ntn_batch500_32_average_losses.pickle', 'wb') as handle:
                pickle.dump(losses, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pbar.set_description(train_info)
        return losses
    def save(self, path):
        torch.save(self._model.state_dict(), path)

    def load_weights(self, path):
        self._model.load_state_dict(torch.load(path))

def main():
    with open('embeddedwords_average_20210912.pickle','rb') as f:
        result=pickle.load(f)
    n_sample=len(result['events'])
    LEARNING_RATE=0.001
    REGULARIZER_WEIGHT=0.0001
    data=result['events'][0:n_sample,:,:]
    dictionary_size=n_sample
    embedding_size=100
    tensor_dim=32
    dropout=0.5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = "cpu"
    #device = torch.device('cpu')
    cuda=True
    #if torch.cuda.is_available(): cuda=True
    batch_size=500
    epochs=50
    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}

    
    model = NeuralTensorNetwork_wrapper(dictionary_size,embedding_size,tensor_dim,dropout,LEARNING_RATE,device,cuda=cuda)
    model.load_weights('./tmp/ntn_trained_32_2_AverageLoss_0.63_Dataleft_0.0.model')
    #eventsembedded=model.predict(data)
    losses=model.fit(data, epochs,REGULARIZER_WEIGHT,dictionary_size,batch_size)
    model.save('./tmp/ntn_trained_32.model')
if __name__ == "__main__":
    main()

