import torch
from transformers import AutoModelForSeq2SeqLM
from preprocessing import Preprocessing

class Predictor():
    def __init__(self):
        self.process = Preprocessing(flag_training=False)
        self.model = self.load_model()

    def load_model(self):
        model = AutoModelForSeq2SeqLM.from_pretrained(
            "/kaggle/working/translator",
            use_safetensors=True,
        )
        return model

    def predict(self, sample):
        self.model.eval()
        inputs = self.process.tokenizer(
            [sample],
            return_tensors="pt"
        )
        with torch.no_grad():
            outputs = self.model.generate(
                                inputs["input_ids"],
                                attention_mask=inputs["attention_mask"],
                                max_length=128,
                              )
        decoded_preds = self.process.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        decoded_preds = [pred.strip() for pred in decoded_preds]
        return decoded_preds


pred = Predictor()
pred.predict("hello everyone")
#bonjour à tous