import pickle
import numpy as np

def load_pickle_file(fileName):
    with open(fileName, 'rb') as f:
        return pickle.load(f)

def load_text_file(filename):
    return [line.strip() for line in open(filename,'r',encoding='utf8').read().strip().split('\n')]



class DataFeeder():

    def __init__(self,text_file , image_file):
        
        texts  = load_text_file(text_file)
        images = load_pickle_file(image_file)

        self.texts , self.images = [],[] 
        for txt , img in zip(texts,images):
            if img.shape[1] < 128:
                continue
            else:
                self.texts.append(txt)
                self.images.append(img[:,:128])

        assert len(self.texts) == len(self.images)
        self.N = len(self.texts)
    
    def sample(self,batch_size):
        indices = np.random.choice(self.N,size=batch_size)
        X,Y = [],[]
        for ind in indices:
            X.append(self.texts[ind])
            Y.append(self.images[ind])
        return X,Y

        

        
        