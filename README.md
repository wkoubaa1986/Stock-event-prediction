# Stock event predition
Due to the size limitation of 25MB per file I wasn't able to put all the collected database and intermediate results.

I put the test database of the scraped news from Reuters as example. the tickerlist used to generate the database is provided.

To scrape the news from Reuters please  use the script:

"crawler_reuters.py"
 
To download the timeseries tocks prices please use:

"crawler_yahoo_finance.py"

To post-process the scraped data use:

"Data_Processing_WEmb.py" ( Note you need to dowload stanford-corenlp-4.2.2 and put it in the same working folder)

## Model scripts:

### Training NTN model:

"model/ntn_pytorch.py"

### Training CNN model :

1- Use "model/prepare_long_term_data.py" to generate the input format for CNN based on the output from NTN model or Data_Processing_WEmb.py 

2-Use "model/train_cnn.py", to train the CNN model

## Trading strategy :

"model/market_simulation.py" 


