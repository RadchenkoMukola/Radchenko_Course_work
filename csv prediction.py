import pandas as pd
import json
from predict import make_prediction_from_text, make_prediction_from_file
from dataset import load_data, tokenize_reviews, create_dataloader
from model import load_model, save_model, load_saved_model
from train import train_model
import argparse

def get_max_scores_with_labels(input_dict, review):
    max_scores = {'review': review}

    categories = set(category.split('_')[0] for category in input_dict.keys() if '_' in category)

    for category in categories:
        max_score = max(input_dict[key] for key in input_dict.keys() if category in key)

        labels = [key.split('_')[1] for key in input_dict.keys() if category in key]
        max_label = [label for label in labels if input_dict[category + '_' + label] == max_score][0]
        if max_score < 0.1:
            max_label = 'neutral'
        max_scores[category] = {
            'score': str(max_score),
            'label': max_label
        }
    sort_list = ['review', 'kitchen', 'service', 'ambience', 'location', 'overall']
    max_scores = {key: max_scores[key] for key in sort_list}

    return max_scores


# Завантаження натренованої моделі
path_to_model = '/data'  # Замініть на шлях до вашої моделі
train_labels = None
model = load_saved_model(path_to_model, train_labels.shape[1] if train_labels is not None else 15)

# Завантаження CSV файлу
df = pd.read_csv('empty_file.csv')

# Створення порожнього списку для зберігання результатів
results = []


# Функція для обробки кожного рядка з колонкою "review"
def process_row(row):
    print(f"Processing row...")
    predictions = make_prediction_from_text(model, str(row['review']))
    predictions = get_max_scores_with_labels(predictions, str(row['review']))
    return predictions



# Проходження кожного рядка та виклик функції для кожного "review"
for index, row in df.iterrows():
    result = process_row(row)
    results.append(result)

# Запис результатів у JSON файл
output_file = 'output.json'
with open(output_file, 'w') as json_file:
    json.dump(results, json_file, indent=4)

print(f"Результати були збережені у файлі: {output_file}")