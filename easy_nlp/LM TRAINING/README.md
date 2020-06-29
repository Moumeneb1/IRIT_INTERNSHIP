# LM Training 

Edit the script args and run it on you model and dataset


```python 

>>>    from Transformers import FlaubertWithLMHeadModel
>>>    from easy_nlp.lm_training import train 
>>>
>>>    args = {
>>>        'train_batch_size': 3,
>>>        'per_gpu_train_batch_size': 8,
>>>        'max_steps': -1,
>>>        'num_train_epochs': 4.0,
>>>        'local_rank': -1,
>>>        'n_gpu': 2,
>>>        'gradient_accumulation_steps': 1,
>>>        'weight_decay': 0.0,
>>>        'learning_rate': 5e-5,
>>>        'adam_epsilon': 1e-8,
>>>        'warmup_steps': 0,
>>>        'seed': 0,
>>>        'mlm': 0.15,
>>>        'max_grad_norm': 1.0,
>>>        'logging_steps': 500,
>>>        'save_steps': 500,
>>>        'evaluate_during_training': True,
>>>        'output_dir': 'flaubert_fine_tuned_alldata',
>>>        'save_total_limit': None,
>>>        'fp16': True,
>>>        'fp16_opt_level': "O1"
>>>    }
>>>    args = pd.Series(args)
>>>    # Load model wih LM Head
>>>    model = FlaubertWithLMHeadModel.from_pretrained("flaubert-base-cased")
>>>    model.cuda()
>>>
>>>    # Load data
>>>    df = load_data('path to csv dataset')
>>>    tokenizer = FlaubertTokenizer.from_pretrained("flaubert-base-cased")
>>>    lines = df["preprocesed_text"].astype(str).values.tolist()
>>>    train_dataset = TextDataset(lines, tokenizer)
>>>    train(args, train_dataset, model, tokenizer)
```