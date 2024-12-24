from datasets import load_dataset, concatenate_datasets, DatasetDict
from configs.translator_config import ConfigDataset, ConfigModel
import torch
class CustomDataset():
    def __init__(self,
                 path_dataset=ConfigDataset.PATH_DATASET,
                 revision=ConfigDataset.REVISION,
                 lang1=ConfigDataset.LANG1,
                 lang2=ConfigDataset.LANG2,
                 train_size=ConfigModel.TRAIN_SIZE,
                 flag_info=True
                ):
        self.raw_data = self.split_dataset(train_size, path_dataset, lang1, lang2)
        self.size = len(self.raw_data["train"])
        if flag_info:
          print("-"*50, "Information of Dataset", "-"*50)
          print(self.raw_data)
          print("-"*50, "Information of Dataset", "-"*50)

    def split_dataset(self, train_size, path_dataset, lang1, lang2):
      raw_data = load_dataset(path_dataset, lang1=lang1, lang2=lang2)
      split_datasets = raw_data["train"].train_test_split(
          train_size=train_size,
          seed=42
      )
      split_datasets["validation"] = split_datasets.pop("test")
      split_datasets = DatasetDict(
          {
              "train": split_datasets["train"].shuffle().select(range(10000)),
              "validation": split_datasets["validation"].shuffle().select(range(1000)),
          }
      )  
      return split_datasets

    def __len__(self):
      return self.size

    def __getitem__(self, index):
      dataset = concatenate_datasets((self.raw_data["train"],
                                      self.raw_data["validation"]))
      data = dataset[index]["translation"]["en"]
      target = dataset[index]["translation"]["fr"]
      return {
          "data_text_en": data,
          "data_label_fr": target
      }

