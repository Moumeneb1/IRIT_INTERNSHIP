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
2. __Preprocess__ : Preprocess Your Data using preprocessing modules using [__preprocessing__](Easy_NLP/preprocessing)


```python 
>>> from preprocessing.text_preprocessing import TextPreprocessing

>>> featuresExtrator = FeaturesExtraction(df,"TEXT")
>>> featuresExtrator.fit_transform()
```


3. __features_extraction__ : extract features from your tweets using features extraction [__feature_extraction__](Easy_NLP/feature_extraction)
4. __training__ : Train your models using our training loops on our predifined models including ,
We enhanced Bert models using several technics : 
-   BERT + LSTM
-   BERT + Custom Sampling 
-   BERT + CNN
-   BERT + ADAPTED ON LM 
-   BERT + FEATURES
-   BERT + CROSS LINGUAL TRAINING 




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
