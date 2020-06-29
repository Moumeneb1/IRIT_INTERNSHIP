from torch import nn
import torch
from transformers import AutoConfig, AutoModel
from torch.nn import CrossEntropyLoss
import sys
import torch.nn.functional as F


class BasicBertForClassification(nn.Module):
    """Bert Model transformer with a classification head on top (a linear layer on top of the pooled output)"""

    def __init__(self, base_bert_model, n_class, dropout_rate=0.1):
        """
        :param bert_model: a bert base model can be Bert,Flaubert or Camembert 
        :param n_class: int
        :param dropout_rate : float 
        """

        super(BasicBertForClassification, self).__init__()
        self.n_class = n_class
        self.bert = base_bert_model
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(self.dropout_rate)
        self.classifier = nn.Linear(
            self.bert.config.hidden_size, self.n_class)

    def forward(self, batch):
        """
        :param batch: batch containing input_ids attention_masks (optional)labels,  
        :return:  (optional) loss : returned when labels provided,pre_softmax : torch.tensor of shape (batch_size, n_class) 
        """
        b_input_ids = batch[0]
        b_input_mask = batch[1]

        outputs = self.bert(input_ids=b_input_ids, attention_mask=b_input_mask)

        # Take the first token hidden state (like Bert)
        outputs = outputs[0]
        pooled_output = outputs[:, 0]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return (logits,)  # Pre-Softmax values

    @staticmethod
    def load(model_path: str):
        """ Load the model from a file.
        @param model_path (str): path to model
        @return model (nn.Module): model with saved parameters
        """
        params = torch.load(
            model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        bert_base_config = args['bert_config']
        bert_base_model = AutoModel.from_config(bert_base_config)
        model = BasicBertForClassification(
            bert_base_model, args['n_class'], args['dropout_rate'])
        model.load_state_dict(params['state_dict'])

        return model

    def save(self, path: str):
        """ Save the model to a file.
        @param path (str): path to the model
        """
        print('save model parameters to [%s]' % path, file=sys.stderr)
        params = {
            'args': dict(bert_config=self.bert.config, n_class=self.n_class, dropout_rate=self.dropout_rate),
            'state_dict': self.state_dict()
        }

        torch.save(params, path)


class BertFeaturesForSequenceClassification(nn.Module):

    def __init__(self, base_bert_model, n_class, n_features, dropout_rate=0.1):
        """
        :param base_bert_model: a bert base model can be Bert,Flaubert or Camembert 
        :param n_class: int
        :pram n_features: int 
        :param dropout_rate : float 
        """

        super(BertFeaturesForSequenceClassification, self).__init__()
        self.n_class = n_class
        self.n_features = n_features
        self.bert = base_bert_model
        self.classifier = nn.Linear(
            self.bert.config.hidden_size + n_features, n_class)

        self.hidden = nn.Linear(
            self.bert.config.hidden_size+self.n_features, self.bert.config.hidden_size+self.n_features)

        self.dropout_rate = dropout_rate

        self.dropout = nn.Dropout(p=self.dropout_rate)

        self.activation = nn.LeakyReLU()
        self.bn = nn.BatchNorm1d(self.bert.config.hidden_size)

    def forward(self, batch):
        """
        :param batch: list[str], list of sentences (NOTE: untokenized, continuous sentences)
        :return: pre_softmax, torch.tensor of shape (batch_size, n_class)
        """
        b_input_ids = batch[0]
        b_input_mask = batch[1]
        b_meta_features = batch[2]

        pooled_output = self.bert(
            input_ids=b_input_ids, attention_mask=b_input_mask)

        output = pooled_output[0]
        pooled_output = output[:, 0]  # Retrieve the first hidden state

        pooled_output = torch.cat([pooled_output, b_meta_features], dim=-1)
        pooled_output = F.gelu(self.hidden(self.dropout(pooled_output)))
        logits = self.classifier(pooled_output)

        return (logits,)  # add hidden states and attention if they are here

    @staticmethod
    def load(model_path: str):
        """ Load the model from a file.
        @param model_path (str): path to model
        @return model (nn.Module): model with saved parameters
        """
        params = torch.load(
            model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        bert_base_config = args['bert_config']
        bert_base_model = AutoModel.from_config(bert_base_config)
        model = BertFeaturesForSequenceClassification(
            bert_base_model, args['n_class'], args['n_features'], args['dropout_rate'])
        model.load_state_dict(params['state_dict'])

        return model

    def save(self, path: str):
        """ Save the model to a file.
        @param path (str): path to the model
        """
        print('save model parameters to [%s]' % path, file=sys.stderr)
        params = {
            'args': dict(bert_config=self.bert.config, n_class=self.n_class, n_features=self.n_features, dropout_rate=self.dropout_rate),
            'state_dict': self.state_dict()
        }

        torch.save(params, path)


class Lstm_Bert_Model(nn.Module):

    def __init__(self, base_bert_model, n_class, dropout_rate=0.1):
        """
        :param base_bert_model: a bert base model can be Bert,Flaubert or Camembert 
        :param n_class: int
        :param dropout_rate : float 

        """

        super(Lstm_Bert_Model, self).__init__()

        self.bert = base_bert_model
        self.lstm_hidden_size = self.bert.config.hidden_size
        self.n_class = n_class
        self.dropout_rate = dropout_rate
        self.lstm = nn.LSTM(self.bert.config.hidden_size,
                            self.lstm_hidden_size, bidirectional=True)
        self.classifier = nn.Linear(
            self.lstm_hidden_size * 2, n_class, bias=True)
        self.dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, batch):
        """
        :param batch: list[str], list of sentences (NOTE: untokenized, continuous sentences)
        :return: pre_softmax, torch.tensor of shape (batch_size, n_class)
        """
        b_input_ids = batch[0]
        b_input_mask = batch[1]

        pooled_output = self.bert(
            input_ids=b_input_ids, attention_mask=b_input_mask)
        output = pooled_output[0]

        output = output.permute(1, 0, 2)
        enc_hiddens, (last_hidden, last_cell) = self.lstm(output)
        # (batch_size, 2*hidden_size)
        output_hidden = torch.cat((last_hidden[0], last_hidden[1]), dim=1)
        output_hidden = self.dropout(output_hidden)
        logits = self.classifier(output_hidden)

        return (logits,)  # add hidden states and attention if they are here

    @staticmethod
    def load(model_path: str):
        """ Load the model from a file.
        @param model_path (str): path to model
        @return model (nn.Module): model with saved parameters
        """
        params = torch.load(
            model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        bert_base_config = args.bert_config
        bert_base_model = AutoModel.from_config(bert_base_config)
        model = Lstm_Bert_Model(
            bert_base_model, args['n_class'], args['dropout_rate'])
        model.load_state_dict(params['state_dict'])

        return model

    def save(self, path: str):
        """ Save the model to a file.
        @param path (str): path to the model
        """
        print('save model parameters to [%s]' % path, file=sys.stderr)
        params = {
            'args': dict(bert_config=self.bert.config, n_class=self.n_class, dropout_rate=self.dropout_rate),
            'state_dict': self.state_dict()
        }

        torch.save(params, path)


class CNN_Bert_Model(nn.Module):

    def __init__(self, base_bert_model, n_class, output_channels, kernel_size, dropout_rate=0.1):
        """
        :param base_bert_model: a bert base model can be Bert,Flaubert or Camembert 
        :param n_class: int
        :pram n_features: int 
        :output_channels : int 
        :kernel_size : int 
        :param dropout_rate : float 

        """

        super(CNN_Bert_Model, self).__init__()

        self.bert = base_bert_model
        self.lstm_hidden_size = self.bert.config.hidden_size
        self.n_class = n_class
        self.dropout_rate = dropout_rate
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.conv1d = m = nn.Conv1d(
            self.bert.config.hidden_size, output_channels, kernel_size)
        #self.hidden_1 = nn.Linear(256,10)
        self.classifier = nn.Linear(output_channels, n_class, bias=True)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.activation = nn.LeakyReLU()

    def forward(self, batch):
        """
        :param batch: list[str], list of sentences (NOTE: untokenized, continuous sentences)
        :return: pre_softmax, torch.tensor of shape (batch_size, n_class)
        """
        b_input_ids = batch[0]
        b_input_mask = batch[1]
        b_meta_features = batch[2]

        pooled_output = self.bert(
            input_ids=b_input_ids, attention_mask=b_input_mask)
        output = pooled_output[0]

        batch_size = output.size(0)
        output = output.permute(0, 2, 1)

        enc_hiddens = self.conv1d(output)
        enc_hiddens = self.activation(enc_hiddens)
        output_hidden, indices = torch.max(enc_hiddens, dim=2)

        output_hidden = self.dropout(output_hidden)
        logits = self.classifier(output_hidden)

        return (logits,)  # add hidden states and attention if they are here

    @staticmethod
    def load(model_path: str):
        """ Load the model from a file.
        @param model_path (str): path to model
        @return model (nn.Module): model with saved parameters
        """
        params = torch.load(
            model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        bert_base_config = args.bert_config
        bert_base_model = AutoModel.from_config(bert_base_config)
        model = CNN_Bert_Model(
            bert_base_model, args['n_class'], args['output_channels'], args['kernel_size'], args['dropout_rate'])
        model.load_state_dict(params['state_dict'])

        return model

    def save(self, path: str):
        """ Save the model to a file.
        @param path (str): path to the model
        """
        print('save model parameters to [%s]' % path, file=sys.stderr)
        params = {
            'args': dict(bert_config=self.bert.config, n_class=self.n_class, output_channels=self.output_channels, kernel_size=self.kernel_size, dropout_rate=self.dropout_rate),
            'state_dict': self.state_dict()
        }

        torch.save(params, path)


class BertMultiTask(nn.Module):

    def __init__(self, config, list_n_class, num_tasks, dropout_rate):
        """
        :param bert_config: str, BERT configuration description
        :param n_class: int
        """

        super(bertMultiTask, self).__init__()
        self.bert = FlaubertModel.from_pretrained(config)
        self.classifiers = []
        self.num_tasks = num_tasks

        for i in range(self.num_tasks):
            self.classifiers.append(nn.Linear(
                self.bert.config.hidden_size, list_n_class[i]))

        self.dropout_rate = dropout_rate

        self.dropout = nn.Dropout(p=self.dropout_rate)

        self.activation = nn.LeakyReLU()
        self.bn = nn.BatchNorm1d(self.bert.config.hidden_size)

    def forward(self, batch):
        """
        :param batch: list[str], list of sentences (NOTE: untokenized, continuous sentences)
        :return: pre_softmax, torch.tensor of shape (batch_size, n_class)
        """
        b_input_ids = batch[0]
        b_input_mask = batch[1]
        b_meta_features = batch[2]

        pooled_output = self.bert(
            input_ids=b_input_ids, attention_mask=b_input_mask)

        #"""Just opn flaubert """
        output = pooled_output[0]
        # take the first token hidden state (like Bert)
        pooled_output = output[:, 0]
        #pooled_output = pooled_output[1]
        #'''the end '''
        # on bert
        logits = []
        for i in self.num_tasks:
            logits.append(self.classifiers[i](self.dropout(pooled_output)))

        #hidden1 = self.linear1(fc_input)
        #hidden2 = self.linear2(self.dropout(hidden1))
        #logits = self.linear3(self.dropout(hidden2))
        # add hidden states and attention if they are here
        outputs = (logits)

        if len(batch) > self.num_tasks+2:

            criterion = CrossEntropyLoss()
            loss = 0
            for i in num_tasks:
                loss += criterion(logits_cat.view(-1,
                                                  list_n_class[i]), batch[self.num_tasks+2+i].view(-1))

            outputs = (loss,) + outputs

        return (logits,)

    @staticmethod
    def load(model_path: str):
        """ Load the model from a file.
        @param model_path (str): path to model
        @return model (nn.Module): model with saved parameters
        """
        params = torch.load(
            model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = NonlinearModel(**args)
        model.load_state_dict(params['state_dict'])

        return model

    def save(self, path: str):
        """ Save the model to a file.
        @param path (str): path to the model
        """
        print('save model parameters to [%s]' % path, file=sys.stderr)

        params = {
            'args': dict(bert_config=self.bert.config, n_class=self.n_class, dropout_rate=self.dropout_rate),
            'state_dict': self.state_dict()
        }

        torch.save(params, path)
