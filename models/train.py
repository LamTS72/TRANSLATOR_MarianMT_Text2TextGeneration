from transformers import (
    get_scheduler,
)
import evaluate
import torch
import os
import numpy as np
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
from huggingface_hub import Repository, HfApi, HfFolder
import math
from torch.nn.utils.rnn import pad_sequence
from configs.translator_config import ConfigModel, ConfigHelper
from translator_model import CustomModel
from preprocessing import Preprocessing
from data.custom_data import CustomDataset

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Used Device: ", device)

class Training():
    def __init__(self, model_name=ConfigModel.MODEL_TOKENIZER,
                 learning_rate=ConfigModel.LEARNING_RATE,
                 epoch=ConfigModel.EPOCHS,
                 num_warmup_steps=ConfigModel.NUM_WARMUP_STEPS,
                 name_metric=ConfigModel.METRICs,
                 path_tensorboard=ConfigModel.PATH_TENSORBOARD,
                 path_save=ConfigModel.PATH_SAVE,
                 dataset=None,
                 process=None
                ):
        self.dataset = dataset
        self.process = process
        self.model = CustomModel(model_name).model
        self.epochs = epoch
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate
        )
        self.lr_scheduler = get_scheduler(
            name="linear",
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=(self.epochs * len(self.process.train_loader))
        )
        self.metric = evaluate.load(name_metric)
        self.writer = SummaryWriter(path_tensorboard)

        # Define necessary variables
        self.api = HfApi(token=ConfigHelper.TOKEN_HF)
        self.repo_name = path_save  # Replace with your repo name
        self.author = ConfigHelper.AUTHOR
        self.repo_id = self.author + "/" + self.repo_name
        self.token = HfFolder.get_token()
        self.repo = self.setup_hf_repo(self.repo_name, self.repo_id, self.token)

    def setup_hf_repo(self, local_dir, repo_id, token):
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)

        try:
            self.api.repo_info(repo_id)
            print(f"Repository {repo_id} exists. Cloning...")
        except Exception as e:
            print(f"Repository {repo_id} does not exist. Creating...")
            self.api.create_repo(repo_id=repo_id, token=token, private=True)

        repo = Repository(local_dir=local_dir, clone_from=repo_id)
        return repo

    def save_and_upload(self, epoch, final_commit=False):
        # Save model, tokenizer, and additional files
        self.model.save_pretrained(self.repo_name)
        self.process.tokenizer.save_pretrained(self.repo_name)

        # Push to Hugging Face Hub
        self.repo.git_add(pattern=".")
        commit_message = "Final Commit: Complete fine-tuned model" if final_commit else f"Epoch {epoch}: Update fine-tuned model and metrics"
        self.repo.git_commit(commit_message)
        self.repo.git_push()

        print(f"Model and files pushed to Hugging Face Hub for epoch {epoch}: {self.repo_id}")

    def compute_metrics(self, eval_preds):
      preds, labels = eval_preds
      # Trong trường hợp mô hình trả về nhiều hơn logit dự đoán
      if isinstance(preds, tuple):
          preds = preds[0]

      decoded_preds = self.process.tokenizer.batch_decode(preds, skip_special_tokens=True)

      # Thay các gía trị -100 trong nhãn vì ta không giải mã chúng
      labels = np.where(labels != -100, labels, self.process.tokenizer.pad_token_id)
      decoded_labels = self.process.tokenizer.batch_decode(labels, skip_special_tokens=True)

      # Thực một một xố hậu xủ lý đơn giản
      decoded_preds = [pred.strip() for pred in decoded_preds]
      decoded_labels = [[label.strip()] for label in decoded_labels]

      result = self.metric.compute(predictions=decoded_preds, references=decoded_labels)
      return {"bleu": result["score"]}

    def postprocess(self, predictions, labels):
      predictions = predictions.cpu().numpy()
      labels = labels.cpu().numpy()

      decoded_preds = self.process.tokenizer.batch_decode(predictions, skip_special_tokens=True)

      # Thay -100 trong nhãn vì ta không thế giải mã chúng.
      labels = np.where(labels != -100, labels, self.process.tokenizer.pad_token_id)
      decoded_labels = self.process.tokenizer.batch_decode(labels, skip_special_tokens=True)

      # Thực hiện một số hậu xử lý đơn giản
      decoded_preds = [pred.strip() for pred in decoded_preds]
      decoded_labels = [[label.strip()] for label in decoded_labels]
      return decoded_preds, decoded_labels

    def fit(self, flag_step=False):
        progress_bar = tqdm(range(self.epochs * len(self.process.train_loader)))
        interval = 200
        for epoch in range(self.epochs):
            # training
            self.model.train()
            n_train_samples = 0
            total_train_loss = 0
            for i, batch in enumerate(self.process.train_loader):
                batch = {k: v.to(device) for k, v in batch.items()}
                n_train_samples += len(batch)
                outputs = self.model.to(device)(**batch)
                losses = outputs.loss
                losses.backward()

                total_train_loss += round(losses.item(),4)

                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                progress_bar.update(1)
                if (i + 1) % interval == 0 and flag_step == True:
                    print("Epoch: {}/{}, Iteration: {}/{}, Train Loss: {}".format(
                        epoch + 1,
                        self.epochs,
                        i + 1,
                        len(self.process.train_loader),
                        losses.item())
                    )
                    self.writer.add_scalar('Train/Loss', round(losses.item(),4), epoch * len(self.process.train_loader) + i)

            # evaluate
            self.model.eval()
            for i, batch in enumerate(self.process.val_loader):
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.no_grad():
                    outputs = self.model.generate(
                                batch["input_ids"],
                                attention_mask=batch["attention_mask"],
                                max_length=128,
                              )

                labels = batch["labels"]
                true_predictions, true_labels = self.postprocess(outputs, labels)
                self.metric.add_batch(predictions=true_predictions, references=true_labels)
                

            epoch_train_loss = total_train_loss / n_train_samples
            print(f"train_loss: {epoch_train_loss}")

            results = self.metric.compute()
            print(f"epoch {epoch}, BLEU score: {results['score']:.2f}")

            # Save and upload after each epoch
            final_commit = ((epoch+1) == self.epochs)
            self.save_and_upload((epoch+1), final_commit)


if __name__ == '__main__':
    dataset = CustomDataset()
    process = Preprocessing(dataset=dataset.raw_data)
    train = Training(dataset=dataset,process=process)
    train.fit()
