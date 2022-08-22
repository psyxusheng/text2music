import os
from pyexpat import model
from torch import nn
from transformers import AlbertConfig , AlbertModel , AlbertTokenizer
from transformers import BertTokenizer



model_path = os.path.join(os.path.dirname(os.path.abspath(__file__) ), 'model_files')

class PretrainedLanguageModel(nn.Module):
    
    def __init__(self,):

        super(PretrainedLanguageModel , self).__init__()
        self.model     = AlbertModel.from_pretrained(model_path)
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
    
    def forward(self,input_sentences):
        output = self.tokenizer(input_sentences,add_special_tokens = True,
                        padding = True,return_tensors = 'pt')
        return self.model(**output,return_dict = True).pooler_output


    

if __name__ == '__main__':
    model  = PretrainedLanguageModel()
    output = model(['今天我要嫁给你','你是不是傻'])
    print(output.size())

