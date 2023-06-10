from transformers import Trainer, TrainingArguments, BertForSequenceClassification, RobertaForSequenceClassification
from setup import train_dataset_bert, dev_dataset_bert, train_dataset_roberta, dev_dataset_roberta

def train_model(model, train_dataset, val_dataset, model_path):
    training_args = TrainingArguments(
        output_dir='./results', 
        num_train_epochs=3,              
        per_device_train_batch_size=16,  
        per_device_eval_batch_size=64,   
        warmup_steps=500,                
        weight_decay=0.01,               
        logging_dir='./logs',
    )

    trainer = Trainer(
        model=model,                       
        args=training_args,                  
        train_dataset=train_dataset,         
        eval_dataset=val_dataset             
    )

    trainer.train()

    # Save the model
    model.save_pretrained(model_path)

if __name__ == "__main__":
    # Initialize models
    bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    roberta_model = RobertaForSequenceClassification.from_pretrained('roberta-base')

    # Call the train_model function for each
    model_path = "./model"
    train_model(bert_model, train_dataset_bert, dev_dataset_bert, f'{model_path}/bert_model')
    train_model(roberta_model, train_dataset_roberta, dev_dataset_roberta, f'{model_path}/roberta_model')
