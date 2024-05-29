import pandas as pd
from predict import make_prediction_from_text, make_prediction_from_file
from dataset import load_data, tokenize_reviews, create_dataloader
from model import load_model, save_model, load_saved_model
from train import train_model
import argparse
import json
import os
from transformers import logging
logging.set_verbosity_error()


# Завантаження вихідного файлу
source_file = "dataset/source_file.csv"
source_data = pd.read_csv(source_file)

# Створення порожнього DataFrame з колонками
columns = [
    'review',
    'service_good', 'service_neutral', 'service_bad',
    'kitchen_neutral', 'kitchen_bad',
    'ambience_good', 'ambience_neutral', 'ambience_bad',
    'location_good', 'location_neutral', 'location_bad',
    'overall_good', 'overall_neutral', 'overall_bad'
]

data = pd.DataFrame(columns=columns)

# Перенесення даних з колонки "translation" до колонки "review" на основі значень колонки "original_locale"
for index, row in source_data.iterrows():
    if row['original_locale'] == 'uk':
        data.loc[index, 'review'] = row['translation']

# Збереження DataFrame у CSV файл
data.to_csv('empty_file.csv', index=False)