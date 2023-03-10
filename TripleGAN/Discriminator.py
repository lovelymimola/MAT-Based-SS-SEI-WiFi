import torch
from torch import nn

class Discriminator(nn.Module):
    def __init__(self,n_label=16,):
        super(Discriminator, self).__init__()
        self.embedding = nn.Embedding(n_label, 1024)
        self.linear1 = nn.Linear(2048,1024)
        self.linear2 = nn.Linear(1024,16)
        self.linear3 = nn.Linear(16,2)

    def forward(self,x,y):
        yembed = self.embedding(y)
        yembed = yembed / torch.norm(yembed, p=2, dim=1, keepdim=True)
        x = torch.cat([x, yembed], dim=1)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)

        return x

if __name__ == "__main__":
    discriminator = Discriminator()
    data_of_data_label = torch.randn((16,1024))
    label_of_data_label = torch.ones((16))

    data_of_gdata_label = torch.randn((16,1024))
    label_of_gdata_label = torch.ones((16))

    data_of_data_clabel = torch.randn((16,1024))
    label_of_data_clabel = torch.ones((16))

    input_data_of_netD = torch.cat([data_of_data_label,data_of_gdata_label,data_of_data_clabel],dim=0)
    input_label_of_netD = torch.cat([label_of_data_label,label_of_gdata_label,label_of_data_clabel],dim=0)
    input_label_of_netD = input_label_of_netD.long()
    x = discriminator(input_data_of_netD, input_label_of_netD)


