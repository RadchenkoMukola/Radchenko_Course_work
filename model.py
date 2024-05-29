from transformers import BertForSequenceClassification
import torch

def load_model(num_labels):
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=num_labels,
        output_attentions=False,
        output_hidden_states=False,
    )
    if torch.cuda.is_available():
        model.cuda()
    return model

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_saved_model(path, num_labels):
    model = load_model(num_labels)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model
