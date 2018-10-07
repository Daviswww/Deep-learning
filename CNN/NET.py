from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2

class Net(nn.Module):
    def __init__(self):
         super(Net, self).__init__()
         self.con1 = nn.Sequential(
                         nn.Conv2d(
                                 #圖片高度
                                 in_channels = 1,
                                 #提取幾個特徵
                                 out_channels = 16,
                                 #捲積核大小
                                 kernel_size = 5,
                                 #掃描一次跳幾格
                                 stride = 1,
                                 #外圍填充(是'SAME'否'VALID')
                                 #padding =(kernel_size-1)/2
                                 padding = 2
                                 ),
                         #線性整流函數
                         nn.ReLU(),
                         #最大池化層(篩選最大值特徵)
                         nn.MaxPool2d(kernel_size = 2)
                         )
         self.con2 = nn.Sequential(
                         nn.Conv2d(16, 32, 5, 1, 2),
                         nn.ReLU(),
                         nn.MaxPool2d(2)
                         )
         #捨去資料
         #self.conv2_drop = nn.Dropout2d()
         self.fc1 = nn.Sequential(
                         nn.Linear(32 * 7 * 7, 120),
                         nn.ReLU()
                         )
         self.fc2 = nn.Sequential(
                         nn.Linear(120, 84),
                         nn.ReLU()
                         )
         self.fc3 = nn.Linear(84, 10)
         
    def forward(self, x):
        x = self.con1(x)
        x = self.con2(x)
        #多維度的張量展平成一維
        #x = x.view(x.size(0), -1) #batch(32 * 7 * 7)
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
#學習
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        #梯度歸零
        optimizer.zero_grad()
        output , last_layer= model(data)
        #計算誤差
        loss = F.nll_loss(output, target)
        #反向傳播
        loss.backward()
        #單次優化
        optimizer.step()
#檢測
def test(args, model, device, test_loader, epoch):
#    繪製分部圖
#    import random
#    try: from sklearn.manifold import TSNE; HAS_SK = True
#    except: HAS_SK = False; print('Please install sklearn for layer visualization')
#    k = random.randrange(0,(len(test_loader.dataset.test_labels) / test_loader.batch_size))
#    
#   把module設置為評估模式
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output , last_layer = model(data)
            #所有批量損失
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            #獲取最大對數概率
            pred = output.max(1, keepdim=True)[1]
            #所有正確量
            correct += pred.eq(target.view_as(pred)).sum().item()
#            if i == k:
#                if HAS_SK:
#                    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
#                    plot_only = 500
#                    low_dim_embs = tsne.fit_transform(last_layer.data.cpu().numpy()[:plot_only, :])
#                    labels = target.data.cpu().numpy()[:plot_only]
#                    plot_with_labels(low_dim_embs, labels)
    test_loss /= len(test_loader.dataset)
    print('Epochs {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
          .format(epoch, test_loss, correct, len(test_loader.dataset),
                  100. * correct / len(test_loader.dataset)))
    
def plot_with_labels(lowDWeights, labels):
    from matplotlib import cm
    plt.cla()
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    for x, y, s in zip(X, Y, labels):
        c = cm.rainbow(int(255 * s / 9)); plt.text(x, y, s, backgroundcolor=c, fontsize=9)
    plt.xlim(X.min(), X.max()); plt.ylim(Y.min(), Y.max()); plt.title('Visualize last layer'); plt.show(); plt.pause(0.01)
    
def input_image(model, image, device):
    model.eval()
    k = image.view(1, 1, 28, 28)
    x, y = model(k.cuda())
#    print(image)
#    print(k.size())    
#    print(x)
    maxn = x.max(1, keepdim=True)[1]
    print('This image is', int(maxn))
def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch_size', type=int, default=640, metavar='N')
    parser.add_argument('--test_batch_size', type=int, default=2000, metavar='N')
    parser.add_argument('--epochs', type=int, default=10, metavar='N')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M')
    parser.add_argument('--no_cuda', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=1, metavar='S')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N')
    #參數
    args = parser.parse_args()
    #為CPU設置種子用於生成隨機數
    torch.manual_seed(args.seed)    
    #GPU運算
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    #轉變型態
    device = torch.device("cuda" if use_cuda else "cpu")    
    kwargs = {'num_workers': 5, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
            batch_size = args.test_batch_size, shuffle=True, **kwargs)
    test_set = datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
    model = Net().to(device)
#    model = torch.load('net1.pkl').to(device)   #讀取
#    plt.ion()    
#============== Train & Test ==============
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas = (0.9, 0.99), eps=1e-04, weight_decay=0)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader, epoch)
    torch.save(model, 'net1.pkl')      #存取
#==========================================
#    plt.ioff()
#============== Test image ==============  
#    img, label = test_set[0]
#    img = cv2.imread('3.jpg')
#    img = img.transpose(2,0,1)
#    img = img[0].transpose(0, 1)
#    img = np.expand_dims(img, axis=0)
#    img = np.expand_dims(img, axis=0)
#    first_train_img = np.reshape(img, (28, 28))
#    plt.matshow(first_train_img, cmap = plt.get_cmap('gray'))
#    plt.show()
#    img = torch.from_numpy(img).float().to(device)
##    print(type(img))
#    input_image(model, img, device)
#==========================================
if __name__ == '__main__':
    main()















