from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch
import pandas as pd
import os

def load_data(path_to_dataset):
    df = pd.read_csv(path_to_dataset)

    reviews = df['review_text'].values
    labels = df.drop(['review_text'], axis=1).values

    train_reviews, val_reviews, train_labels, val_labels = train_test_split(reviews, labels, test_size=0.1, random_state=42)
    return train_reviews, val_reviews, train_labels, val_labels

def tokenize_reviews(reviews):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    input_ids = []
    attention_masks = []

    for review in reviews:
        encoded_dict = tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=64,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    return input_ids, attention_masks

def create_dataloader(input_ids, attention_masks, labels, batch_size, for_training=True):
    labels = torch.tensor(labels)
    dataset = TensorDataset(input_ids, attention_masks, labels)
    if for_training:
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    return dataloader
