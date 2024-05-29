from transformers import get_linear_schedule_with_warmup
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW
import torch


def train_model(model, train_dataloader, validation_dataloader, epochs):
    optimizer = AdamW(model.parameters(),
                      lr=2e-5,
                      eps=1e-8)

    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    loss_func = BCEWithLogitsLoss()

    for epoch_i in range(0, epochs):
        total_train_loss = 0

        model.train()

        for step, batch in enumerate(train_dataloader):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad()

            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

            loss = loss_func(outputs.logits, b_labels.type_as(outputs.logits))
            total_train_loss += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_dataloader)

        model.eval()

        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0

        for batch in validation_dataloader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            with torch.no_grad():
                outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

            loss = loss_func(outputs.logits, b_labels.type_as(outputs.logits))
            total_eval_loss += loss.item()

        avg_val_loss = total_eval_loss / len(validation_dataloader)

        print("Epoch number: ", epoch_i + 1,
              "Training Lost: ", avg_train_loss,
              "Validation Lost: ", avg_val_loss)

    return model
