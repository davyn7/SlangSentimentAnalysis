import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast, RobertaTokenizerFast
import torch

# Load the data
df = pd.read_csv('data/combined.csv')

# Initial split: 60% for training, 40% for a temporary set
df_train, df_temp = train_test_split(df, train_size=0.6)

# Split the temporary set evenly into the testing and development sets
df_dev, df_test = train_test_split(df_temp, train_size=0.5)

# Split into training and testing sets
train_texts, dev_texts, test_texts, train_labels, dev_labels, test_labels = df_train['text'], df_dev['text'], df_test['text'], df_train['sentiment'], df_dev['sentiment'], df_test['sentiment']

# Initialize tokenizers
bert_tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
roberta_tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')

# Tokenize the texts
train_encodings_bert = bert_tokenizer(list(train_texts), truncation=True, padding=True)
dev_encodings_bert = bert_tokenizer(list(dev_texts), truncation=True, padding=True)
test_encodings_bert = bert_tokenizer(list(test_texts), truncation=True, padding=True)

train_encodings_roberta = roberta_tokenizer(list(train_texts), truncation=True, padding=True)
dev_encodings_roberta = roberta_tokenizer(list(dev_texts), truncation=True, padding=True)
test_encodings_roberta = roberta_tokenizer(list(test_texts), truncation=True, padding=True)

# Create a dataloader
class TwitterDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Create training and testing datasets
train_dataset_bert = TwitterDataset(train_encodings_bert, train_labels)
dev_dataset_bert = TwitterDataset(dev_encodings_bert, dev_labels)
test_dataset_bert = TwitterDataset(test_encodings_bert, test_labels)

train_dataset_roberta = TwitterDataset(train_encodings_roberta, train_labels)
dev_dataset_roberta = TwitterDataset(dev_encodings_roberta, dev_labels)
test_dataset_roberta = TwitterDataset(test_encodings_roberta, test_labels)