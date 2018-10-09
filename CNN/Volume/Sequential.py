import torch
from torch.autograd import Variable #變量
import torch.nn.functional as F
import matplotlib.pyplot as plt
#torch.unsqueeze 一維數據變二維數據
n_data = torch.ones(100, 2)
x0 = torch.normal(2*n_data, 1)      # class0 x data (tensor), shape=(100, 2)
y0 = torch.zeros(100)               # class0 y data (tensor), shape=(100, 1)
x1 = torch.normal(-2*n_data, 1)     # class1 x data (tensor), shape=(100, 2)
y1 = torch.ones(100)                # class1 y data (tensor), shape=(100, 1)
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # shape (200, 2) FloatTensor = 32-bit floating
y = torch.cat((y0, y1), ).type(torch.LongTensor)    # shape (200,) LongTensor = 64-bit integer

# torch can only train on Variable, so convert them to Variable
x, y = Variable(x), Variable(y)

plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
plt.show()
#methon 1
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
        x = F.relu(self.hidden(x))      # 激勵函數(神經元)
        x = self.predict(x)             # 預測值
        return x
#搭圖 神經網路(輸入幾個, 幾個神經元, 輸出幾個特徵)
net1 = Net(2, 10, 2)
print(net1)
#methon 1
net2 = torch.nn.Sequential(     #累積神經層
    torch.nn.Linear(2, 10),     #第一層
    torch.nn.ReLU(),            #激勵層(類別) 與 relu效果相同
    torch.nn.Linear(10, 2)      #第二層
)
print(net2)
#可視化
plt.ion()
plt.show()
#優化神經網路(神經網路所有參數, 學習效率)
optimizer = torch.optim.SGD(net.parameters(), lr = 0.02) 
#計算誤差
#loss_func = torch.nn.MSELoss() #用均方差計算(回歸問題)
loss_func = torch.nn.CrossEntropyLoss() #概率計算(分類問題) 每個點有不同的概率做分類

for t in range(100):
    out = net(x)                 # input x and predict based on x
    loss = loss_func(out, y)     # must be (1. nn output, 2. target), the target label is NOT one-hotted

    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()        # apply gradients
    
    if t % 10 == 0 or t in [3, 6]:
        # plot and show learning process
        plt.cla()
        #概率輸出 F.softmax(out) [0.1, 0.2, 0.7]
        _, prediction = torch.max(F.softmax(out), 1)
        pred_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = sum(pred_y == target_y)/200.
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color':  'red'})
        plt.show()
        plt.pause(0.1)

plt.ioff()