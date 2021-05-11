from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import logging
from transformers import *
import torch

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

train_path = './train_file.csv'
eval_path = './eval_file.csv'

train_df = pd.read_csv(train_path)

eval_df = pd.read_csv(eval_path)

model_args = ClassificationArgs()
model_args.num_train_epochs=1
model_args.labels_list = ["1", "2", "3", "4", "5"]

model = ClassificationModel(
    "bert", "DeepPavlov/rubert-base-cased", args=model_args
)

# Train
model.train_model(train_df)

# Predict
mname = "./model_sent_rubert"
model_sent = AutoModelForSequenceClassification.from_pretrained(mname)
tokenizer_sent = BertTokenizerFast.from_pretrained(mname)

text = ''  # input
@torch.no_grad()
def predict(text):
		inputs = tokenizer_sent(text, max_length=512, padding=True, truncation=True, return_tensors='pt')
		outputs = model_sent(**inputs)
		predicted = torch.nn.functional.softmax(outputs.logits, dim=1)
		predicted = torch.argmax(predicted, dim=1).numpy()
		return predicted
            
ans = predict(text)[0]
print(ans)


