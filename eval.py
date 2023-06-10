from transformers import Trainer, BertForSequenceClassification, RobertaForSequenceClassification
from setup import test_dataset_bert, test_dataset_roberta
from train import bert_model, roberta_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def evaluate_model(model, dataset):
    trainer = Trainer(model, compute_metrics=compute_metrics)
    result = trainer.evaluate(dataset)
    return result

if __name__ == "__main__":
    # Initialize models
    bert_model = BertForSequenceClassification.from_pretrained('models/bert_model')
    roberta_model = RobertaForSequenceClassification.from_pretrained('models/roberta_model')

    # Call the evaluate_model function
    bert_result = evaluate_model(bert_model, test_dataset_bert)
    roberta_result = evaluate_model(roberta_model, test_dataset_roberta)

    print("BERT Evaluation Result: ", bert_result)
    print("RoBERTa Evaluation Result: ", roberta_result)