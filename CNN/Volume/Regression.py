import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
#一維數據變二維數據
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1) 
y = x.pow(2) + 0.2*torch.rand(x.size()) 
#神經網路只能輸入變量
x, y = Variable(x), Variable(y)

plt.scatter(x.data.numpy(), y.data.numpy())
plt.show()

class Net(torch.nn.Module):
    #搭建神經層所需的訊息
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        #神經層模塊
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # 隱藏層(輸入, 輸出)
        self.predict = torch.nn.Linear(n_hidden, n_output)   # 預測神經層(接收的神經元, 輸出)
    #神經網路前向傳遞的過程
    def forward(self, x):
        #搭建神經層
        x = F.relu(self.hidden(x))      # 機率函數(神經元)
        x = self.predict(x)             # 預測
        return x
#搭圖
net = Net(1, 10, 1)
#可視化
plt.ion()
plt.show()
#優化神經網路(神經網路所有參數, 學習效率)
optimizer = torch.optim.SGD(net.parameters(), lr = 0.5) 
#計算誤差
loss_func = torch.nn.MSELoss() #用均方差計算

for t in range(100):
    #輸入訊息
    prediction = net(x)
    #計算誤差
    loss = loss_func(prediction, y)     # 誤差函式(預測值, 真實質)
    #把所有參數parameters梯度降為0
    optimizer.zero_grad()
    #反向傳遞
    loss.backward()
    #優化梯度
    optimizer.step()
    #每五步輸出一次
    if t % 5 == 0:  
        # plot and show learning process
        plt.cla()
        #原始程度
        plt.scatter(x.data.numpy(), y.data.numpy()) 
        #神經網路學習到的程度
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        #打印誤差
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color':  'red'})
        plt.show()
        plt.pause(0.1)

plt.ioff()