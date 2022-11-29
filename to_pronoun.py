from crosslingual_coreference import Predictor


class change_to_noun():
  def __init__(self):
  # choose minilm for speed/memory and info_xlm for accuracy
    self.predictor = Predictor(
        language="en_core_web_sm", device=-1, model_name="spanbert"
    )
  
  def input_txt(self, sample_txt):
    text = (sample_txt)   
    #print(self.predictor.predict(text)["resolved_text"])
    return self.predictor.predict(text)["resolved_text"]


