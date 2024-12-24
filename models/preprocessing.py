from transformers import(
    AutoTokenizer,
    DataCollatorForSeq2Seq
)
from torch.utils.data import DataLoader
from configs.translator_config import ConfigModel

class Preprocessing():
    def __init__(self, model_tokenizer=ConfigModel.MODEL_TOKENIZER,
                 batch_size=ConfigModel.BATCH_SIZE,
                 max_input_length=ConfigModel.MAX_INPUT_LENGTH,
                 max_target_length=ConfigModel.MAX_TARGET_LENGTH,
                 model=None,
                 dataset=None,
                 flag_training=True):
      self.tokenizer = AutoTokenizer.from_pretrained(model_tokenizer)
      self.data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer,
                                                           model=model)
      if flag_training:
        print("-"*50, "Information of Tokenizer", "-"*50)
        print(self.tokenizer)
        print("-"*50, "Information of Tokenizer", "-"*50)
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.tokenized_dataset = self.map_tokenize_dataset(dataset)
        self.train_loader, self.val_loader = self.data_loader(batch_size)

    def tokenize_dataset(self, sample):
      inputs = [ex["en"] for ex in sample["translation"] ]
      targets = [ex["fr"] for ex in sample["translation"] ]
      tokenized_input = self.tokenizer(
          inputs,
          max_length=self.max_input_length,
          truncation=True
      )
      with self.tokenizer.as_target_tokenizer():
        labels = self.tokenizer(
          targets,
          max_length=self.max_target_length,
          truncation=True
      )
      tokenized_input["labels"] = labels["input_ids"]
      return tokenized_input

    def map_tokenize_dataset(self, dataset):
      tokenized_dataset = dataset.map(
          self.tokenize_dataset,
          batched=True,
          remove_columns=dataset["train"].column_names
      )
      tokenized_dataset.save_to_disk("/kaggle/working/data")
      return tokenized_dataset


    def data_loader(self, batch_size):
      train_loader = DataLoader(
        self.tokenized_dataset["train"],
        shuffle=True,
        batch_size=batch_size,
        collate_fn=self.data_collator,
      )
      val_loader = DataLoader(
        self.tokenized_dataset["validation"],
        batch_size=batch_size,
        collate_fn=self.data_collator,
      )

      return train_loader, val_loader

