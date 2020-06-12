# Easy NLP 


## Structure

- Feature Extractions : This package contains feature extraction methods
- Models : This package contains different usable models architectures 
- preprocessing : This Folder contains Text preprocessing methods 
- Scrappping : This contains Scrapping methods  
- Training : This Folder contains training loops and methods directly usable 
- LM TRAINING : This folder contains a builtin script that you can call to adapt bert base models to your dataset
- Examples : This Folder contains example on how we used the Pipeline on Crisis classification and Tweet stigmatisation  .
- DataVisualisation : On this Folder we show examples on how we uses Data vizualisation and transformation to improve our understanding of the problem 



## How to use

This project has a main pipeline with several sideline pieces of analyisis. The steps of the main flow are:

1. __Scrapping data__ : You can eaither scrap tweets or enhance existing tweets features, Just with tweets IDS using [__feature_extraction__](Easy_NLP/scrapping)
2. __Preprocess__ : Preprocess Your Data using preprocessing modules using [__preprocessing__](Easy_NLP/preprocessing)
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
- The __Grid-search__ of hyperparameters for the Deep Learning models using SKORCH to wrap . [Code](Codes/deep_learning/1_design/1_basic_models_gridsearch.ipynb)
- The analysis of the __class distribution__. [Code](Codes/preprocessing/3_class_distribution.Rmd)
- The __app__ for displaying LDA and embeddings. [Code](Codes/app/). Note: Unifinished because we didn't publish the embeddings.
- The analysis of the __information gain__. [Code](Codes/machine_learning/1_design/info_gain.Rmd)
  
  
## How to re-use on different projects

This is not library, and hence there is no plug-and-play for reusing the code. In order to reuse the code and ideas in differnt projects, the best approach would be to re-use the snippets of code descripted above. The vast majority of the code was made using python classes, that can be found in the folders called _classes_ or _libraries_ in the _.py_ files. This is easily reuseable code. In R a big part of the code was made using functions, defined within the scripts. In the python notebook there are also definitions for the deep learning models that can be easily re-use.

## Reference

--
