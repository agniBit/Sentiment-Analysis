import numpy as np
from torch import device, tensor
from transformers import BertTokenizer, get_linear_schedule_with_warmup, BertForSequenceClassification, AdamW, BertConfig
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, RandomSampler,SequentialSampler, DataLoader
import pandas as pd
class Dataloaders:
    def __init__(self, batch_size=32, epochs=4, MAX_LEN = 46):
        self.batch_size = batch_size
        self.MAX_LEN = MAX_LEN

    def load_data_from_csv(self):
        dataset = pd.read_csv('./processed_data.csv')
        labels = [0 if t == 'negative' else 1 for t in dataset['airline_sentiment'].values]
        return dataset,labels

    def tokenize_and_pad_data(self, dataset):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        input_token_seq = []
        for sent in dataset['text'].values:
            encoded_sent = tokenizer.encode(str(sent), add_special_tokens=True)
            input_token_seq.append(encoded_sent)
        input_token_seq = pad_sequences(input_token_seq, maxlen=self.MAX_LEN, dtype="long",
                                  value=0, truncating="post", padding="post")
        return input_token_seq

    def get_dataloaders(self):
        dataset, labels = self.load_data_from_csv()
        input_token_seq = self.tokenize_and_pad_data(dataset)
        # create attention masks
        attention_masks = []
        # mask = 1 for token 0 for padding
        for seq in input_token_seq:
            att_mask = [int(token_id > 0) for token_id in seq]
            attention_masks.append(att_mask)
        # train test split data
        train_inputs, val_inputs, train_labels, val_labels = \
            train_test_split(input_token_seq, labels, random_state=42, test_size=0.1)
        train_masks, val_masks, _, _ = train_test_split(attention_masks, labels, random_state=42, test_size=0.1)
        # Create the DataLoader for our training set
        train_data = TensorDataset(tensor(train_inputs), tensor(train_masks), tensor(train_labels))
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.batch_size)
        # Create the DataLoader for val set.
        val_data = TensorDataset(tensor(val_inputs), tensor(val_masks), tensor(val_labels))
        val_sampler = SequentialSampler(val_data)
        val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=self.batch_size)
        return train_dataloader,val_dataloader









