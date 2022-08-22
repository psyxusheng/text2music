import torch
from torch import nn
from torch.nn import functional as F

class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()
        # Custom Implementation because the Voice Conversion Cycle GAN
        # paper assumes GLU won't reduce the dimension of tensor by 2.

    def forward(self, input):
        return input * torch.sigmoid(input)


class PixelShuffle(nn.Module):
    def __init__(self, upscale_factor):
        super(PixelShuffle, self).__init__()
        # Custom Implementation because PyTorch PixelShuffle requires,
        # 4D input. Whereas, in this case we have have 3D array
        self.upscale_factor = upscale_factor

    def forward(self, input):
        n = input.shape[0]
        c_out = input.shape[1] // 2
        w_new = input.shape[2] * 2
        return input.view(n, c_out, w_new)

def upSample(in_channels, out_channels, kernel_size, stride, padding):
    convLayer = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=padding),
                              nn.PixelShuffle(upscale_factor=2),
                              nn.InstanceNorm2d(num_features=out_channels // 4,
                                                affine=True),
                              GLU())
    return convLayer

class ResidualLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,stride=1, padding=1):
        super(ResidualLayer, self).__init__()

        self.conv1d_layer = nn.Sequential(nn.Conv1d(in_channels  = in_channels,
                                                    out_channels = out_channels,
                                                    kernel_size  = kernel_size,
                                                    stride=stride , 
                                                    padding=padding),
                                          nn.InstanceNorm1d(num_features=out_channels,
                                                            affine=True))

        self.conv_layer_gates = nn.Sequential(nn.Conv1d(in_channels=in_channels,
                                                        out_channels=out_channels,
                                                        kernel_size=kernel_size,
                                                        stride=stride,
                                                        padding=padding),
                                              nn.InstanceNorm1d(num_features=out_channels,
                                                                affine=True))

        self.conv1d_out_layer = nn.Sequential(nn.Conv1d(in_channels=out_channels,
                                                        out_channels=in_channels,
                                                        kernel_size=kernel_size,
                                                        stride=stride,
                                                        padding=padding),
                                              nn.InstanceNorm1d(num_features=in_channels,
                                                                affine=True))

    def forward(self, input):
        h1_norm = self.conv1d_layer(input)
        h1_gates_norm = self.conv_layer_gates(input)

        # GLU
        h1_glu = h1_norm * torch.sigmoid(h1_gates_norm)

        h2_norm = self.conv1d_out_layer(h1_glu)
        return input + h2_norm

class T2Song(nn.Module):
    def __init__(self ,) -> None:

        super(T2Song , self).__init__()

        self._tograph = nn.Sequential(nn.Linear(768,512*16),
                                      nn.Tanh())

        self._up1 = upSample(512,1024,5,1,2)
        self._up2 = upSample(256,512,5,1,2)
        self._up3 = upSample(128,128,5,1,2)
        self._up4 = upSample(32,32,5,1,2)
        self._up5 = upSample(8,4,5,1,2)

    def forward(self, inputs):
        # nonzero
        batch_size = inputs.size(0)
        init_graph = self._tograph(inputs)
        init_graph = init_graph.view(batch_size,512,4,4)
        
        vec        = self._up1(init_graph)
        vec        = self._up2(vec)
        vec        = self._up3(vec)
        vec        = self._up4(vec)
        vec        = self._up5(vec).squeeze(1)

        return vec 


if __name__ == '__main__':
    model = T2Song()
    torch.save(model.state_dict(),'./saved_model.bin')

    """
    texts = torch.randint(0,1000,[13,23])

    outputs = model(texts)

    print(outputs.size())
    """


        


        