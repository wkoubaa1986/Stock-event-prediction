# Trading strategy:
import numpy as np
import pickle
import datetime
import torch
import json
from cnn_pytorch import LMS_CNN_wrapper
  

with open('../input/stockPrices_raw.json') as json_file:
    stock_coll=json.load(json_file)

def new_date(sample_date,Ndays):
    t=datetime.datetime.strptime(sample_date,'%Y%m%d')+datetime.timedelta(days=Ndays)
    return t.strftime('%Y%m%d')

def extract_price_data(sample_date,stock_coll,company):
    future_price_data=list()
    day_futur=2
    iday=1
    while iday<day_futur and iday<20:# get next day price

        try:
            future_price_data=(stock_coll[company]['open'][new_date(sample_date,iday)],stock_coll[company]['high'][new_date(sample_date,iday)],stock_coll[company]['low'][new_date(sample_date,iday)],stock_coll[company]['close'][new_date(sample_date,iday)])
        except:
            day_futur+=1
            pass
        iday+=1
    return future_price_data
def long_strategy(future_price_data):
    output=future_price_data[3]/future_price_data[0]
    if future_price_data[1]/future_price_data[0]>=1.02:
        output=future_price_data[1]/future_price_data[0]
    return output
def short_strategy(future_price_data):
    output=future_price_data[3]/future_price_data[0]
    if future_price_data[2]/future_price_data[0]<=0.98:
        output=future_price_data[2]/future_price_data[0]
    return output

with open('./input_cnn/cnn_input_Test.pickle','rb') as f:
    result=pickle.load(f)
 

data=result['long_term_mean']
batch_size=len(data)
dataframe=result['dataframe']


data=np.expand_dims(data, axis=1)
data=torch.from_numpy(np.double(data)).float()
test_label=np.array([max(0.,float(x)) for x in dataframe[:,2].astype(np.float)])
test_label=torch.from_numpy(np.double(test_label)).long ()
model = LMS_CNN_wrapper()

model.load_weights('./tmp/cnn_tensor(0.5671).model')
test_set = torch.utils.data.TensorDataset(data, test_label)
test_loader = torch.utils.data.DataLoader(
			dataset=test_set,
			batch_size=batch_size)
outputs,acc=model.test(test_loader)
prediction=model.predict(test_loader)
prediction=prediction.numpy()

Benefit=0
Benefit_rand=0
Benefit_AAPL=0
Benefit_FB=0
Benefit_GOOGL=0
trading_position=100
ind_AAPL=np.where(dataframe[:,0]=='AAPL')
ind_AAPL=ind_AAPL[0]
data_AAPL=dataframe[ind_AAPL,:]
ind_FB=np.where(dataframe[:,0]=='FB')
ind_FB=ind_FB[0]
data_FB=dataframe[ind_FB,:]
ind_GOOGL=np.where(dataframe[:,0]=='GOOGL')
ind_GOOGL=ind_GOOGL[0]
data_GOOGL=dataframe[ind_GOOGL,:]
correct=0
for inews in range(0,len(dataframe)):
    date=dataframe[inews,1]
    company=dataframe[inews,0]
    future_price_data=extract_price_data(date,stock_coll,company)
    pred=prediction[inews]
    if pred==0:
        prc=short_strategy(future_price_data)
        benefit_i=(1/prc-1)*trading_position
    else:
        prc=long_strategy(future_price_data)
        benefit_i=(prc-1)*trading_position
    Benefit+=benefit_i
    if company=='AAPL':
        Benefit_AAPL+=benefit_i
    if company=='FB':
        Benefit_FB+=benefit_i
    if company=='GOOGL':
        Benefit_GOOGL+=benefit_i
    if np.random.randint(0,2)==0:
        if dataframe[inews,2]=='-1.0':
            correct+=1
        prc=short_strategy(future_price_data)
        benefit_i=(1/prc-1)*trading_position
    else:
        if dataframe[inews,2]=='1.0':
            correct+=1
        prc=long_strategy(future_price_data)
        benefit_i=(prc-1)*trading_position
    Benefit_rand+=benefit_i
        
    
        
        