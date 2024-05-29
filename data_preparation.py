import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
import torch
import os


def load_data(file_path):
    df = pd.read_csv(file_path)
    reviews = df['review_text'].values
    labels = df.drop(['review_text'], axis=1).values
    return reviews, labels


def split_data(reviews, labels, test_size=0.1, random_state=42):
    return train_test_split(reviews, labels, test_size=test_size, random_state=random_state)


def prepare_data(reviews, tokenizer, max_length=64):
    input_ids = []
    attention_masks = []

    for review in reviews:
        encoded_dict = tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=max_length,
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
    dataset = TensorDataset(input_ids, attention_masks, labels)
    if for_training:
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)

    return DataLoader(dataset, sampler=sampler, batch_size=batch_size)
