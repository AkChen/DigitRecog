import torch as t
import torchvision
from CNN_MNIST import CNN as CNN_M
from resnet_image import resnet18 as CNN_S
from CNN_Fusion_2 import FusionNet as CNN_F
from BioModalDataset import BioModalDataset
from torch.utils.data import DataLoader,TensorDataset
import torch.optim as optim
import datetime
from torchvision.datasets import mnist,svhn
import argparse
import sys
data_root_path = './data/'
train_batch_size = 64
test_batch_size = 64

def net_train(net,data_loader,opt,loss_func,cur_e,args):

    net.train()
    begin_time = datetime.datetime.now()

    train_loss = 0.0
    batch_num = int(len(data_loader.dataset) / args.batch_size)

    for i,data in enumerate(data_loader,0):
        #print('batch:%d/%d' % (i,batch_num))
        m,v,l = data
        m,v,l = m.type(t.FloatTensor).cuda(),v.type(t.FloatTensor).cuda(),l.type(t.LongTensor).cuda()

        opt.zero_grad()

        output = net(m,v)
        loss = loss_func(output,l)
        loss.backward()
        opt.step()

        # loss
        train_loss += loss

    end_time = datetime.datetime.now()
    delta_time = (end_time-begin_time)
    delta_seconds = (delta_time.seconds*1000 + delta_time.microseconds)/1000

    print('epoch:%d loss:%.4f time:%.4f'% (cur_e,train_loss.cpu(),(delta_seconds)))


def net_test(net,data_loader):
    num = len(data_loader.dataset)
    correct = 0
    net.eval()
    with t.no_grad():
        for i, data in enumerate(data_loader, 0):
            m, v, l = data
            m, v, l = m.type(t.FloatTensor).cuda(), v.type(t.FloatTensor).cuda(), l.type(t.LongTensor).cuda()

            output = net(m,v)

            predict_label = t.argmax(output,dim=1)
            correct += (predict_label == l).sum()


    return correct.cpu().numpy()*1.0/num



def main():
    parser = argparse.ArgumentParser(description='AID_PRETRAIN')
    parser.add_argument('--batch_size', type=int, default=128,help='training batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epoch',type=int,default=20,help='training epoch')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help = 'SGD momentum (default: 0.9)')
    args = parser.parse_args()

    print('loading data')

    train_dataset = BioModalDataset(file='./data/biomodal/biomodal_train.npy')
    test_dataset = BioModalDataset(file='./data/biomodal/biomodal_test.npy')

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)


    mnist_net = CNN_M(10)
    mnist_net_cuda = t.nn.DataParallel(mnist_net,device_ids=[0]).cuda()
    svhn_net = CNN_S(10)
    svhn_net_cuda = t.nn.DataParallel(svhn_net,device_ids=[0]).cuda()
    fusion_net = CNN_F(mnist_net_cuda,svhn_net_cuda,10)
    fuson_net_cuda = t.nn.DataParallel(fusion_net,device_ids=[0]).cuda()
    loss_func = t.nn.CrossEntropyLoss()

    optimizer = optim.SGD(params=fuson_net_cuda.parameters(), lr=args.learning_rate, momentum=args.momentum)
    max_acc = 0.0
    print('training mnist svhn with CNN Resnet' )
    for e in range(args.epoch):
        net_train(fuson_net_cuda,train_dataloader,optimizer,loss_func,e,args)
        test_acc = net_test(fuson_net_cuda,test_dataloader)
        print('EPOCH:%d TEST ACC:%f' % (e,test_acc))
        sys.stdout.flush()
        if test_acc > max_acc:
            max_acc = test_acc

    f = open('./result/train_mnist_svhn_on_CNN.txt',mode='w')
    f.write('max_acc:%.4f' % (max_acc))
    f.close()

if __name__ == '__main__':
    main()


