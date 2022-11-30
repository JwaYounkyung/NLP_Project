import torch
import transformers
from transformers import PegasusForConditionalGeneration, PegasusTokenizer


class paraphrase():
  def __init__(self):
  # choose minilm for speed/memory and info_xlm for accuracy

    self.model_name = 'tuner007/pegasus_paraphrase'
    #torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.tokenizer = PegasusTokenizer.from_pretrained(self.model_name)
    self.model = PegasusForConditionalGeneration.from_pretrained(self.model_name)
  
  def get_response(self,input_text,num_return_sequences=1,num_beams=10):
    batch = self.tokenizer([input_text],truncation=True,padding='longest',max_length=60, return_tensors="pt")
    translated = self.model.generate(**batch,max_length=60,num_beams=num_beams, num_return_sequences=num_return_sequences, temperature=1.5)
    tgt_text = self.tokenizer.batch_decode(translated, skip_special_tokens=True)
    return tgt_text