
import json
import datetime
import re
import pandas as pd
from gensim.models import Word2Vec
import numpy as np
import string
import functools
import spacy
from pycorenlp import StanfordCoreNLP
import time
from multiprocessing.dummy import Pool
import pickle
from functools import partial



#****************************function********************************

def corenlp_extractor(text,debug=False,server=0):
    svo_triples=[]
    output=stanford_server_list[server].annotate(text, properties={
        'annotators': 'openie',
        'outputFormat': 'json'
        })
    if type(output)==dict:
        for sentence in output['sentences']:
            svo_triples.extend(list(map(lambda x: (x['subject'],x['relation'],x['object']),sentence['openie'])))
    return svo_triples

def to_SVO_embedding(text,word_embedding):
    return np.array([ word_embedding[str(term)] for term in list(text) ]).mean(axis=0)

def core_nlp_to_SVO_embedding(text,word_embedding):
    embeddings=[ word_embedding.wv[str(term)] if str(term) in word_embedding.wv else None for term in list(nlp(text))]
    embeddings=list(filter(lambda x: x is not None,embeddings))
    if len(embeddings)==0:
        return None
    return np.array(embeddings).mean(axis=0)

def clean_str(string):
    
    string = re.sub(r"[^A-Za-z0-9().,!?\'\`]", " ", string)
    string = re.sub(r"(\s)+", " ", string)
    string = re.sub(r"UPDATE(\s)*[0-9]*", "", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def new_date(sample_date,Ndays):
    t=datetime.datetime.strptime(sample_date,'%Y%m%d')+datetime.timedelta(days=Ndays)
    return t.strftime('%Y%m%d')

def extract_price_data(sample_date,stock_coll,company):
    future_price_data=list()
    day_futur=2
    day_past=7
    iday=0
    while iday<day_futur and iday<20:# get next day price

        try:
            future_price_data.append(stock_coll[new_date(sample_date,iday)])
        except:
            day_futur+=1
            pass
        iday+=1
    past_price_data=list()
    iday=0
    while iday<day_past and iday<20:

        try:
            past_price_data.append(stock_coll[new_date(sample_date,-iday)])
        except:
            day_past+=1
            pass
            
        iday+=1
    past_price_data=pd.DataFrame(past_price_data)
    return (future_price_data,past_price_data)
    



def extract_all_events_for_company(sample_server,stock_coll,word_embedding,events_extractor=corenlp_extractor,**kwargs):
    processed_data=[]
    sample=sample_server[0]
    server=sample_server[1]
    sample_date=str(sample['date'])
    text=clean_str(sample['content'])
    company=sample['symbol']
    processed_data=None
    try:
        adjclose=stock_coll[company]['adjClose']
    except:
        return processed_data
    
    future_price_data, past_price_da=extract_price_data(sample_date,adjclose,company)
    if len(future_price_data)<2:return processed_data
    price_label=np.sign(future_price_data[1]-future_price_data[0])
    svo_triples=events_extractor(text,server=server,**kwargs)

    if len(svo_triples) == 0:
        return processed_data
    svo_result=[]
    for svo in svo_triples:
        if len(svo) == 0:
            continue
        o1=svo[0]
        p=svo[1]
        o2=svo[2]
        svo_embedding=[core_nlp_to_SVO_embedding(o1,word_embedding),core_nlp_to_SVO_embedding(p,word_embedding),core_nlp_to_SVO_embedding(o2,word_embedding)]
        if svo_embedding[0] is None or svo_embedding[1] is None or svo_embedding[2] is None:
            continue
        svo_result.append(np.array(svo_embedding))
        
        
    
    if len(svo_result)>0:
        processed_data=[np.array(svo_result),svo_triples,price_label,company,sample_date]
        print(company+' processed '+sample_date)
    return processed_data

nlp = spacy.load('en_core_web_lg')
#Launch the CoreNLP server
#os.chdir('C:/Users/WKOUBAA/Documents/MscProject/Event-Driven-Stock-Prediction/stanford-corenlp-4.2.2')
#os.system('java -mx6g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -timeout 5000')
#os.chdir('C:/Users/WKOUBAA/Documents/MscProject/Event-Driven-Stock-Prediction')


##********************************************************
Test=True
stanford=StanfordCoreNLP('http://127.0.0.1:9000')
stanford_server_list=[stanford] 
#Loading data
if Test:
    samples=pd.read_csv('./input/news_reuters_test.csv',low_memory=False)
    #samples=pd.read_csv('./input/news_reuters_original.csv',low_memory=False)
else: 
    samples=pd.read_csv('./input/news_reuters.csv',low_memory=False)

samples_2=list()
for i in range(0,len(samples)):
    ele=dict()
    for j in range(0,len(samples.columns)):
        if j==3:
            try:
                ele['content']=samples[samples.columns[j]][i]+' . '+samples[samples.columns[j+1]][i]
            except:
                ele['content']=samples[samples.columns[j]][i]
                    
        elif j==4:
            pass
        else:
            ele[samples.columns[j]]=samples[samples.columns[j]][i]
    samples_2.append(ele)

with open('./input/stockPrices_raw.json') as json_file:
    stock_coll=json.load(json_file)   

sample_dates=samples.sort_values('date')
sample_dates=sample_dates.date



samples_2=samples_2
server_index=(list(range(len(stanford_server_list)))*len(samples_2))[:len(samples_2)]   
sample_server=list(zip(samples_2,server_index))

def process_sample_to_sentences(sample):
    print(sample['symbol'] + ' :' + str(sample['date'])+'\n')
    return list(map(lambda x: str(x),list(nlp(clean_str(sample['content']))) ))


start=time.time()
pool=Pool(20)
sentences=pool.map(process_sample_to_sentences,samples_2)
end = time.time()
print(end - start)

SENTENCES_PATH = './input/sentences_test.json'
WORDEMB_PATH = './input/wordembedding_20210912.model'

with open(SENTENCES_PATH,'w') as outfile:
    json.dump(sentences,outfile)

if Test:
    word_embedding=Word2Vec.load(WORDEMB_PATH)
else:
    start=time.time()
    word_embedding=Word2Vec(sentences,min_count=1,vector_size=100,workers=4,sg=1)
    end = time.time()
    print(end - start)
    
    word_embedding.save(WORDEMB_PATH)

start=time.time()
pool=Pool(20)
partial_work = partial(extract_all_events_for_company,stock_coll=stock_coll,word_embedding=word_embedding,events_extractor=corenlp_extractor) 
processed_data = pool.map(partial_work, sample_server)
end = time.time()
print(end - start)

final_result=list(filter(lambda x: x is not None, processed_data))
svo_embeddings=tuple([ result[0] for result in final_result])
label_events=[ result[2]  for result in final_result for event in result[0]]
dates_events=[ result[4]  for result in final_result for event in result[0]]
comp_events=[ result[3]  for result in final_result for event in result[0]]
events=np.vstack(tup=svo_embeddings)
total_events={'comp_events':comp_events,'dates_events':dates_events,'events':events,'label_events':label_events}

# Averaged tupple embedding per valid news 
svo_embeddings_avr=tuple([np.expand_dims(np.vstack ((np.mean(np.vstack(result[0][:,0,:]),axis=0),np.mean(np.vstack(result[0][:,1,:]),axis=0),np.mean(np.vstack(result[0][:,2,:]),axis=0))),axis=0) for result in final_result])
label_events_avr=[ result[2]  for result in final_result ]
dates_events_avr=[ result[4]  for result in final_result ]
comp_events_avr=[ result[3]  for result in final_result ]
events_avr=np.vstack(tup=svo_embeddings_avr)
total_events_avr={'comp_events':comp_events_avr,'dates_events':dates_events_avr,'events':events_avr,'label_events':label_events_avr}


with open('./model/embeddedwords_original_database.pickle', 'wb') as handle:
    pickle.dump(total_events, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('./input/final_result_original_database.pickle', 'wb') as handle:
    pickle.dump(final_result, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('./model/embeddedwords_average_original_database.pickle', 'wb') as handle:
    pickle.dump(total_events_avr, handle, protocol=pickle.HIGHEST_PROTOCOL)




