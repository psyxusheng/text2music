import torch
from torch import nn
from albertmodel import PretrainedLanguageModel
from T2I import T2Song

from DataFeeder import DataFeeder

DF  = DataFeeder('cache/texts.txt','cache/coded_sps_norm.pickle')
PLM = PretrainedLanguageModel()
T2S = T2Song()

opt_plm = torch.optim.Adam(PLM.parameters() , lr = 1e-5)
opt_gen = torch.optim.Adam(T2S.parameters() , lr = 1e-3)
loss_func = nn.MSELoss()

for i in range(1000):

    x,y = DF.sample(13)
    y_tensor = torch.tensor(y).float()
    vecs = PLM(x)
    fake = T2S(vecs)
    loss = loss_func(fake,y_tensor)

    opt_plm.zero_grad()
    opt_gen.zero_grad()

    loss.backward()

    opt_plm.step()
    opt_gen.step()  

    if i%10 == 0:
        print(loss.data.numpy())