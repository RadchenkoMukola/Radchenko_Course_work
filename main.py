
from predict import make_prediction_from_text, make_prediction_from_file
from dataset import load_data, tokenize_reviews, create_dataloader
from model import load_model, save_model, load_saved_model
from train import train_model
import argparse
import json
import os
from transformers import logging
logging.set_verbosity_error()


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


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train or make predictions with BERT model")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'predict'],
                        help='Mode to run the script in')
    parser.add_argument('--data', type=str, required=False, help='Path to the dataset', default='./dataset/dataset.csv')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--model', type=str, default=None, help='Path to the saved model')
    parser.add_argument('--review', type=str, default=None, help='Review text to predict')
    parser.add_argument('--predict_file', type=str, default=None, help='Path to the file with reviews for prediction')
    parser.add_argument('--epochs', type=str, default=4, help='Amount of epochs to train the model for')
    return parser.parse_args()


def main():
    args = parse_arguments()

    if args.mode == 'train' or args.predict_file is not None:
        train_reviews, val_reviews, train_labels, val_labels = load_data(args.data)
        train_input_ids, train_attention_masks = tokenize_reviews(train_reviews)
        val_input_ids, val_attention_masks = tokenize_reviews(val_reviews)
        train_dataloader = create_dataloader(train_input_ids, train_attention_masks, train_labels, args.batch_size)
        val_dataloader = create_dataloader(val_input_ids, val_attention_masks, val_labels, args.batch_size,
                                           for_training=False)
    else:
        train_labels = None
        val_dataloader = None

    if args.mode == 'train':
        model = load_model(train_labels.shape[1])
        model = train_model(model, train_dataloader, val_dataloader, args.epochs)
        save_model(model, args.model)
    else:
        if args.model:
            model = load_saved_model(args.model, train_labels.shape[1] if train_labels is not None else 15)
            if args.review:
                predictions = make_prediction_from_text(model, args.review)
                predictions = get_max_scores_with_labels(predictions, args.review)
                print(json.dumps(predictions, indent=4))
            elif args.predict_file:
                # TODO: need to refactor this
                current_dir = os.path.dirname(os.path.abspath(__file__))
                predict_file = open(os.path.join(current_dir, args.predict_file), 'r')
                predict_file.readline()
                for line in predict_file:
                    predictions = make_prediction_from_text(model, line)
                    predictions = get_max_scores_with_labels(predictions, line)
                    print(json.dumps(predictions, indent=4))
                predict_file.close()
                # predictions = make_prediction_from_file(model, args.predict_file, args.batch_size)
            else:
                print("Please provide a review text or a path to the prediction file.")
                return




if __name__ == "__main__":
    main()
