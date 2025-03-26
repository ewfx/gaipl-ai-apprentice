import torch
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer
)
from datasets import Dataset
from typing import Dict

class BERTClassifier:
    def __init__(self, model_name: str = "bert-base-uncased"):
        self.model_name = model_name
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=10,
            problem_type="multi_label_classification"
        )

    def train(self, train_dataset: Dataset, val_dataset: Dataset):
        """Fine-tune BERT model"""
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )

        trainer.train()

    def predict(self, text: str) -> Dict[str, float]:
        """Predict regulatory categories"""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        outputs = self.model(**inputs)
        probabilities = torch.nn.functional.sigmoid(outputs.logits)

        return {
            "Basel III": probabilities[0][0].item(),
            "MiFID II": probabilities[0][1].item(),
            "Dodd-Frank": probabilities[0][2].item(),
        }
