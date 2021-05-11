import nltk
import torch
import string
import pandas as pd
from transformers import *

nltk.download('punkt')

mname = "./model_sent_rubert"
model_sent = AutoModelForSequenceClassification.from_pretrained(mname)
tokenizer_sent = BertTokenizerFast.from_pretrained(mname)

text = ''  # input

div = ''
text_list = nltk.sent_tokenize(text)
tt = str.maketrans(dict.fromkeys(string.punctuation))
if len(text.split()) > 1:
  for text_f in text_list:
    if len(text_f.split()) > 1:
      lb = []
      text_f = text_f.translate(tt)
      input_ids = tokenizer_sent.encode(text_f, return_tensors="pt")

      with torch.no_grad():
          out = model_sent.base_model.forward(input_ids=input_ids, output_attentions=True, 
                                        return_dict=True, use_cache=False)

      cross = torch.stack(out.attentions).mean(0)
      mean_attention = cross.mean(dim=1).squeeze(0).numpy()
      mean_attention[:, -1] = 0
      mean_attention = (mean_attention.T / mean_attention.sum(axis=1)).T

      df = pd.DataFrame(mean_attention)

      di = {}
   
      for i in range(len(tokenizer_sent.encode(text_f))):
        if i != 0 and i != len(tokenizer_sent.encode(text_f)) - 1:
          di[tokenizer_sent.tokenize(text_f)[i - 1]] = df[i].sum()

      list_d = list(di.items())
      list_d.sort(key=lambda i: i[1], reverse=True)
      count = len(nltk.word_tokenize(text_f)) // 3

      t_l = text_f.split()

      l = []
      for i in list_d[:3]:
        for k in t_l:

          if i[0].replace('#', '') in k and len(i[0].replace('#', '')) > 1:
            l.append(k.lower().capitalize())

      lb.append(set(l))

      for z in lb:

        if len(z) > 1:
          pred_div = ', '.join(z)
          s2_l = pred_div.lower().split(', ')
          s1_l = text_f.lower().split()
          d = {}
          for i in range(len(s2_l)):
            for k in range(len(s1_l)):
              if s2_l[i] in s1_l[k]:
                d[i] = k
          list_dd = list(d.items())
          list_dd.sort(key=lambda i: i[1])

          div_word = [i for i in s2_l]
          for e, i in enumerate(list_dd):
            div_word[e] = s2_l[i[0]].capitalize()

          div += ', '.join(div_word) + '. '
    else:
      div += ''
else:
  div = 'Предложение слишком короткое для поиска ключевых слов'

div_final = '<center><p style="color: black"><font color="#e253dd">Ключевые слова: </font>{}</p></center>'.format(div)
