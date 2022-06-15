import numpy as np
import pickle
import datetime
import time



small_database=True
if small_database:
    data_path='./input_cnn/cnn_input_Test.pickle'
    # with open('embeddedwords_average_original_database.pickle','rb') as f:
    #        result1=pickle.load(f)
    with open('embeddedwords_average_test.pickle','rb') as f:
           result2=pickle.load(f)
       #result=result1
       #n_sample=len(result['events'])
    ntn_news_embedding_or=result2['events']
    result={'comp_name':result2['comp_events'],'dates_events':result2['dates_events'],'label_events':result2['label_events']}
else:
        
    data_path='./input_cnn/cnn_input_BIG.pickle'
    with open('embeddedwords_average_original_database.pickle','rb') as f:
           result1=pickle.load(f)
    with open('embeddedwords_average_20210912.pickle','rb') as f:
           result2=pickle.load(f)
       #result=result1
       #n_sample=len(result['events'])
    ntn_news_embedding_or=np.concatenate((result1['events'],result2['events']),axis=0)
    result={'comp_name':result1['comp_events']+result2['comp_events'],'dates_events':result1['dates_events']+result2['dates_events'],'label_events':result1['label_events']+result2['label_events']}
#*******************************************************************************
# Avearging event tuple per day
# ntn_news_embedding_or=np.expand_dims(ntn_news_embedding_or, axis=1)
Nsample=len(result['comp_name'])
ntn_news_embedding_or=ntn_news_embedding_or[0:Nsample,:,:]
dataframe=np.array([np.asarray(result['comp_name'][0:Nsample]),np.asarray(result['dates_events'][0:Nsample]),np.asarray(result['label_events'][0:Nsample])])
dataframe=np.rot90(dataframe)
dataframe_sort=np.lexsort(( dataframe[:,0],dataframe[:,1]))
companies=dataframe[:,0]
dates_original=dataframe[:,1]
label_original=dataframe[:,2]
comp_list=np.sort(companies)
comp_list=np.unique(companies)
comp_name=None
dates_events=None
label_events=None
ntn_news_embedding=None
count_c=-1
count=-1
start=time.time()
for company in comp_list:
    count_c+=1
    ind_c=np.where(companies==company)[0]
    dates=dates_original[ind_c]
    dates=np.sort(dates)
    dates=np.unique(dates)

    for date in dates:
        count+=1
        indx_date=list(set.intersection(set(np.where(companies==company)[0]) , set(np.where(dates_original==date)[0])))
        label_unique=label_original[indx_date]
        label_unique=np.unique(label_unique)
        labeli=np.array([label_unique[0]])
        if 0 in labeli:
            labeli=np.array([1.])
        if len(label_unique)!=1:
            if (1 in label_unique) and (-1 in label_unique):
                print('wrong input :'+company+': '+date)
            for i in  label_unique:
                if i != 0: labeli=np.array([i]);break                   

        if label_events is None:
            label_events=labeli
        else:
            label_events=np.concatenate((label_events,labeli))
        
        if comp_name is None:
            comp_name=np.array([company])
        else:
            comp_name=np.concatenate((comp_name,np.array([company])))
            
        if ntn_news_embedding is None:
            ntn_news_embedding=np.expand_dims(np.mean(ntn_news_embedding_or[indx_date,:],axis=0),axis=0)
        else:
            ntn_news_embedding=np.vstack((ntn_news_embedding,np.expand_dims(np.mean(ntn_news_embedding_or[indx_date,:],axis=0),axis=0)))
    if dates_events is None:
        dates_events=dates
    else:
        dates_events=np.concatenate((dates_events,dates))

    if count % 100 ==0:
            end = time.time()    
            print( 'Processed {} % in {} seconds time spent'.format(float(round(count_c / len(comp_list)*100,2)),float(round(end-start,2))))
dataframe= np.transpose(np.asarray([comp_name,dates_events,label_events]))#{'comp_name':comp_name,'dates_events':dates_events,'label_events':label_events}  
# with open('./input_cnn/cnn_intermi_BIG.pickle', 'rb') as f:
#     result=pickle.load(f)
#ntn_news_embedding=result['ntn_news_embedding']
#dataframe=result['dataframe']
ntn_mean=np.mean(ntn_news_embedding,axis=1)
ntn_concat=np.hstack((ntn_news_embedding[:,0,:],ntn_news_embedding[:,1,:],ntn_news_embedding[:,2,:]))
# ntn_concat=np.mean(ntn_news_embedding,axis=1)
print('Averaging input finished')
# with open('./input_cnn/cnn_intermi_BIG.pickle', 'wb') as handle:
#             pickle.dump({"ntn_news_embedding":ntn_news_embedding,'dataframe':dataframe}, handle, protocol=pickle.HIGHEST_PROTOCOL)
#**************************************************************************************************************************
#long_term=None
companies=dataframe[:,0].copy()
comp_list=np.sort(companies)
comp_list=np.unique(companies)

dates_original=dataframe[:,1].copy()
ntn_mean_sort=ntn_mean.copy()
ntn_concat_sort=ntn_concat.copy()
# long_term_avr=np.zeros((ntn_news_embedding.shape[0],30,ntn_news_embedding.shape[1]))
long_term_mean=np.zeros((ntn_mean.shape[0],30,ntn_mean.shape[1]))
long_term_concat=np.zeros((ntn_concat.shape[0],30,ntn_concat.shape[1]))
sparsity_long_term=np.zeros((ntn_news_embedding.shape[0],1))
sparsity_mid_term=np.zeros((ntn_news_embedding.shape[0],1))
n_comp=len(comp_list)
count=-1
count_c=-1
start=time.time()
indx_row_start=0
for company in comp_list:
    count_c+=1
    ind_c=np.where(companies==company)[0]
    dates=dates_original[ind_c]
    dates=np.unique(dates)
    dates=np.sort(dates)
    for date in dates:
        count+=1
        indx_row_start=count
        indx=list(set.intersection(set(np.where(companies==company)[0]) , set(np.where(dates_original==date)[0])))

    # long term window 30 days
        #short_termi=self.ntn_news_embedding[indx,:]
        d=datetime.datetime.strptime(date,"%Y%m%d")
        date_list=[(d- datetime.timedelta(days=idays)).strftime("%Y%m%d") for idays in range(0,30)]
        iday_count=-1
        spari_long_term=0
        spari_mid_term=0
        for iday in date_list:
            iday_count+=1
            indx=list(set(np.where(companies==company)[0]) & set(np.where(dates_original==iday)[0]))
           
            if len(indx)==1:
                spari_long_term+=1
                if iday_count<7:
                    spari_mid_term+=1
                long_term_mean[indx_row_start,29-iday_count,:]=ntn_mean_sort[indx,:]
                long_term_concat[indx_row_start,29-iday_count,:]=ntn_concat_sort[indx,:]
            elif len(indx)==0:
                #short_termi=np.vstack((np.array([np.zeros(EMBEDDING_DIM)]),short_termi))
                pass
            else:
                print('More that event for '+company+' :'+iday)
                pass
        sparsity_long_term[indx_row_start]=spari_long_term/30
        sparsity_mid_term[indx_row_start]=spari_mid_term/7

    if count_c % 100 ==0:
        end = time.time()    
        print( 'Processed {} % in {} seconds time spent'.format(float(round(count_c / len(comp_list)*100,2)),float(round(end-start,2))))

    ntn_mean_sort=np.delete(ntn_mean_sort,ind_c,0)
    ntn_concat_sort=np.delete(ntn_concat_sort,ind_c,0)
    dates_original=np.delete(dates_original,ind_c,0)
    companies=np.delete(companies,ind_c,0)
dataframe=np.array([np.asarray(dataframe[:,0]),np.asarray(dataframe[:,1]),np.asarray(dataframe[:,2]),sparsity_mid_term[:,0],sparsity_long_term[:,0]])
dataframe=np.transpose(dataframe)
with open(data_path, 'wb') as handle:
            pickle.dump({"long_term_mean": long_term_mean,"long_term_concat": long_term_concat,"ntn_news_embedding":ntn_news_embedding,'dataframe':dataframe}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    

with open(data_path, 'rb') as handle:
    final_result=pickle.load(handle)
