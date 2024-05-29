import torch
from torch.nn import Softmax
import pandas as pd
from transformers import BertTokenizer
from dataset import tokenize_reviews, create_dataloader

class_names = [
    "service_good", "service_neutral", "service_bad", "kitchen_good", "kitchen_neutral", "kitchen_bad", "ambience_good",
    "ambience_neutral", "ambience_bad", "location_good", "location_neutral", "location_bad", "overall_good",
    "overall_neutral", "overall_bad"
]


def make_prediction_from_text(model, text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    softmax = Softmax(dim=1)

    # Tokenize the review
    encoded_review = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=64,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = encoded_review['input_ids'].to(device)
    attention_mask = encoded_review['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, token_type_ids=None, attention_mask=attention_mask)

    logits = outputs[0]
    predictions = softmax(logits)
    predictions = predictions.detach().cpu().numpy()

    # Create a dictionary mapping class names to probabilities
    output = {class_names[i]: predictions[0, i] for i in range(len(class_names))}

    return output


def make_prediction_from_file(model, file_path, batch_size):
    reviews = pd.read_csv(file_path)['review_text'].values
    input_ids, attention_masks = tokenize_reviews(reviews)
    dataloader = create_dataloader(input_ids, attention_masks, None, batch_size, for_training=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    predictions = []
    softmax = Softmax(dim=1)

    for batch in dataloader:
        b_input_ids, b_input_mask = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

        logits = outputs.logits
        logits = logits.detach().cpu().numpy()
        predictions.append(logits)

    predictions = softmax(torch.tensor(predictions))
    return predictions


def make_prediction(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    predictions = []
    softmax = Softmax(dim=1)

    for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask = batch

        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        predictions.append(logits)

    predictions = softmax(torch.tensor(predictions))
    return predictions
