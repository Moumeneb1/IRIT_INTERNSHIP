# Easy NLP 


## Structure


The project is structred like this : 


- Feature Extractions : This package contains feature extraction methods
- Models : This package contains different usable models architectures 
- preprocessing : This Folder contains Text preprocessing methods 
- Scrappping : This contains Scrapping methods  
- Training : This Folder contains training loops and methods directly usable 
- LM TRAINING : This folder contains a builtin script that you can call to adapt bert base models to your dataset
- Examples : This Folder contains example on how we used the Pipeline on Crisis classification and Tweet stigmatisation  .
- DataVisualisation : On this Folder we show examples on how we uses Data vizualisation and transformation to improve our understanding of the problem 

```
.
├── ...
├── EasyNLP
│   ├── Feature Extractions
│   ├── Models
│   ├── preprocessing
│   ├── Scrappping
│   ├── Training 
│   ├── LM TRAINING
│   ├── Examples 
|   └── DataVisualisation
├── setup.py
└── ...
```





## How to use

This project has a main pipeline with several sideline pieces of analyisis. The steps of the main flow are:

1. __Scrapping data__ : You can eaither scrap tweets or enhance existing tweets features, Just with tweets IDS using [__feature_extraction__](Easy_NLP/scrapping)

```python 
>>> from EasyNLP.scrapping.twitter_scrap import Scrapper

>>> scrapper = Scrapper()
>>> keywords= ['innondation','occitanie','crises']
>>> lang = ['fr']
>>> begindate = datetime.datetime(2020, 5, 17)
>>> enddate = datetime.datetime.now()
>>> limit = 150
>>> scrapper.get_tweets_df(keywords,lang,begindate,enddate,limit)
```

2. __Preprocess__ : Preprocess Your Data using preprocessing modules using [__preprocessing__](Easy_NLP/preprocessing)


```python 
>>> from EasyNLP.preprocessing.text_preprocessing import TextPreprocessing
>>> text_preprocessing = TextPreprocessing(df,"TEXT")
>>> text_preprocessing.fit_transform()
```




3. __features_extraction__ : extract features from your tweets using features extraction [__feature_extraction__](Easy_NLP/feature_extraction)

```python 
>>> from EasyNLP.preprocessing.feature_enginering import FeaturesExtraction
>>> featuresExtrator = FeaturesExtraction(df,"TEXT")
>>> featuresExtrator.fit_transform()
```

The package also allows tokenizing texts as Bert input, the Bert model requires Mask and tokens as IDS 

```python
from EasyNLP.feature_extraction.bert_input import BertInput
from Transformers import FlaubertTokenizer
Tokenizer = FlaubertTokenizer.from_pretrained('flaubert-base-cased')
bert_input= BertInput(Tokenizer)
X_train = bert_input.fit_transform(sentences_train)
```


4. __training__ : Train your models using our training loops on our predifined models including ,
We enhanced Bert models using several technics : 
-   BERT + LSTM
-   BERT + Custom Sampling 
-   BERT + CNN
-   BERT + ADAPTED ON LM 
-   BERT + FEATURES
-   BERT + CROSS LINGUAL TRAINING 

```python 
>>> from transformers import FlaubertModel
>>> from EAasyNLP.models.models import BasicBertForClassification,
>>> base_model = FlaubertModel.from_pretrained('flaubert-base-cased')
>>> model = BasicBertForClassification(base_model,n_class)
```


```python
>>> from transformers import AdamW,get_linear_schedule_with_warmup
>>> from EasyNLP.training.train import train

>>> optimizer = AdamW(model.parameters(),
                  lr = 1e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )
>>> scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = 100)

>>> # Number of training epochs (authors recommend between 2 and 4)
>>> epochs = 5

>>> # Total number of training steps is number of batches * number of epochs.
>>> total_steps = len(train_dataloader) * epochs 

>>> # Create the learning rate scheduler.
>>> scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)


>>> train(model,train_dataloader,validation_dataloader,epochs,torch.device('cuda'),optimizer,scheduler,criterion)
```


As side projects we have

- The training of our own __BERT_ADAPTED_LM__,[Code](Codes/deep_learning/1_design/2_word_embedding_train.ipynb)
- The __Vizualisation of training__ , using TensoarBoardX 
- Training of ULMFIT using FastAI
- Exploring [LASER](https://engineering.fb.com/ai-research/laser-multilingual-sentence-embeddings/) with synthetic DATA Augmentation.

  
## INSTALL 
```bash
$ pip install easy_NLP
```

or 

```bash
$ git clone https://github.com/Moumeneb1/IRIT_INTERNSHIP.git
$ cd IRIT_INTERNSHIP
$ pip install .
```


## How to re-use on different projects

This library,is intended for NLP entheusiast to start quickly working on classification problems without struggling with common NLP pipeline modules. This out of the box pipeline modules will help you build a baseline solution quickly that you can improve with proper fineTuning .

You can look at the examples folder to know how to use IT 

## Reference

--
