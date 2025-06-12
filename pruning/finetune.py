import logging
import os
from dataclasses import dataclass
from typing import Dict, Optional

import evaluate
import numpy as np
from datasets import load_dataset, concatenate_datasets, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    model_name: str = "/home/jovyan/shares/SR004.nfs2/amaksimova/models/cultura_1.3b"
    dataset_names: Dict[str, str] = None
    output_dir: str = "models/summarization_model"
    learning_rate: float = 2e-6
    batch_size: int = 4
    num_epochs: int = 1
    max_input_length: int = 1024
    max_target_length: int = 128
    test_size: float = 0.2
    val_size: float = 0.5
    seed: int = 42
    push_to_hub: bool = False
    use_fp16: bool = True
    prefix: str = "summarize: "

    def __post_init__(self):
        if self.dataset_names is None:
            self.dataset_names = {
                "xlsum": ("csebuetnlp/xlsum", "russian"),
                "mixed": "RussianNLP/Mixed-Summarization-Dataset"
            }


class SummarizationPipeline:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.data_collator = None
        self.metrics = None

    def initialize_components(self):
        """Initialize model, tokenizer and metrics"""
        logger.info("Initializing model and tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.config.model_name)
        self.data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.config.model_name
        )
        self.metrics = evaluate.load("rouge")

    def load_and_prepare_datasets(self) -> DatasetDict:
        """Load and preprocess datasets"""
        logger.info("Loading and preparing datasets")
        
        datasets = []
        for name, spec in self.config.dataset_names.items():
            try:
                if isinstance(spec, tuple):
                    dataset = load_dataset(spec[0], spec[1])
                else:
                    dataset = load_dataset(spec)
                
                prepared = self._prepare_dataset(dataset)
                datasets.append(prepared)
                logger.info(f"Loaded dataset: {name} with {len(prepared)} samples")
            except Exception as e:
                logger.error(f"Failed to load dataset {name}: {str(e)}")
                raise

        combined = concatenate_datasets(datasets).shuffle(seed=self.config.seed)
        
        # Split dataset
        train_test = combined.train_test_split(test_size=self.config.test_size)
        test_valid = train_test["test"].train_test_split(
            test_size=self.config.val_size
        )
        
        final_dataset = DatasetDict({
            "train": train_test["train"],
            "validation": test_valid["train"],
            "test": test_valid["test"]
        })
        
        logger.info(
            f"Dataset sizes - Train: {len(final_dataset['train'])}, "
            f"Val: {len(final_dataset['validation'])}, "
            f"Test: {len(final_dataset['test'])}"
        )
        
        return final_dataset

    def _prepare_dataset(self, dataset):
        """Prepare individual dataset"""
        if isinstance(dataset, DatasetDict) and "train" in dataset:
            dataset = dataset["train"]
        
        # Remove unnecessary columns
        columns_to_keep = {"text", "summary"}
        columns_to_remove = [
            col for col in dataset.column_names 
            if col not in columns_to_keep
        ]
        return dataset.remove_columns(columns_to_remove)

    def preprocess_function(self, examples):
        """Tokenize and prepare model inputs"""
        inputs = [self.config.prefix + doc for doc in examples["text"]]
        model_inputs = self.tokenizer(
            inputs,
            max_length=self.config.max_input_length,
            truncation=True,
            padding="max_length"
        )
        
        labels = self.tokenizer(
            text_target=examples["summary"],
            max_length=self.config.max_target_length,
            truncation=True,
            padding="max_length"
        )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def compute_metrics(self, eval_pred):
        """Compute ROUGE metrics"""
        predictions, labels = eval_pred
        decoded_preds = self.tokenizer.batch_decode(
            predictions,
            skip_special_tokens=True
        )
        
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(
            labels,
            skip_special_tokens=True
        )
        
        result = self.metrics.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True
        )
        
        prediction_lens = [
            np.count_nonzero(pred != self.tokenizer.pad_token_id)
            for pred in predictions
        ]
        result["gen_len"] = np.mean(prediction_lens)
        
        return {k: round(v, 4) for k, v in result.items()}

    def train(self):
        """Run the training pipeline"""
        self.initialize_components()
        dataset = self.load_and_prepare_datasets()
        
        tokenized_dataset = dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=["text", "summary"]
        )
        
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.config.output_dir,
            eval_strategy="epoch",
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=self.config.num_epochs,
            predict_with_generate=True,
            fp16=self.config.use_fp16,
            push_to_hub=self.config.push_to_hub,
            logging_dir=os.path.join(self.config.output_dir, "logs"),
            report_to="tensorboard",
            load_best_model_at_end=True,
            metric_for_best_model="rougeL",
            greater_is_better=True,
            save_strategy="epoch",
        )
        
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
        )
        
        logger.info("Starting training")
        train_result = trainer.train()
        trainer.save_model()
        
        logger.info("Evaluating on test set")
        test_metrics = trainer.evaluate(tokenized_dataset["test"])
        logger.info(f"Test metrics: {test_metrics}")
        
        return train_result


if __name__ == "__main__":
    config = TrainingConfig(
        output_dir="./summarization_model",
        num_epochs=1,
        push_to_hub=False
    )
    
    pipeline = SummarizationPipeline(config)
    pipeline.train()