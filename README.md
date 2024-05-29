# REVAN 1.0

## Description
The main script, main.py, utilizes a BERT-based model to predict the sentiment and category of a given review. The model has been trained on a dataset of restaurant reviews and is capable of providing insights into the quality of the kitchen, service, ambience, and location aspects of a restaurant based on customer feedback.

## Getting Started
These instructions will guide you on how to execute the script on your local machine for training and prediction purposes.

### Prerequisites
Install the required packages listed in requirements.txt by running the following command:
```commandline
pip install -r requirements.txt
```
### Running the script
The script is designed to be run from the command line. You can either train a new model or make predictions using a pre-trained model. To execute the script, use the following command:
```commandline
python main.py --mode <mode> [options]
```
Replace <mode> with either train to train a new model, or predict to make predictions with an existing model.

The available options are as follows:
- `--data <path>`: The path to the dataset. Default is './dataset/dataset.csv'.
- `--batch_size <int>`: The batch size for training. Default is 32.
- `--model <path>`: The path to the saved model. This argument is mandatory when in 'predict' mode.
- `--review <text>`: The review text to predict. This argument is used only in 'predict' mode.
- `--predict_file <path>`: The path to the file with reviews for prediction. This argument is used only in 'predict' mode.
- `--epochs <int>`: The number of epochs to train the model for. Default is 4. This argument is used only in 'train' mode.

## Usage Examples

### Training a new model
If you dont have a prepared dataset, or want to create a new one, you have to prepare a csv file of the next format:
```csv
"review_text","service_good","service_neutral","service_bad","kitchen_good","kitchen_neutral","kitchen_bad","ambience_good","ambience_neutral","ambience_bad","location_good","location_neutral","location_bad","overall_good","overall_neutral","overall_bad"
```
To train the model, run the following command:
```commandline
python3 main.py --mode train --data /path_to_your_dataset --model /path_to_save_model
```
### Making predictions
To make predictions, you would use the predict mode. Here is an example:
```commandline
python3 main.py --mode predict --review "your review here" --model /path_to_your_model
```
You can also make predictions from a file:
```commandline
python3 --mode predict --predict_file /path_to_your_reviews_file --model /path_to_your_model
```

### Output
The output of the script is a json response with the following format:
```json
{
    "review": "We love it because they always work diligently, the taste and service are excellent. I recommend it.\n",
    "kitchen": {
        "score": "0.32000488",
        "label": "good"
    },
    "service": {
        "score": "0.12155116",
        "label": "good"
    },
    "ambience": {
        "score": "0.04841398",
        "label": "neutral"
    },
    "location": {
        "score": "0.09103607",
        "label": "neutral"
    },
    "overall": {
        "score": "0.3335524",
        "label": "good"
    }
}
```

