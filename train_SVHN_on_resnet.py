import torch as t
import torchvision
from resnet_image import resnet101
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import datetime
from torchvision.datasets import mnist, svhn
import argparse
import sys
data_root_path = './data/SVHN/'


def net_train(net, data_loader, opt, loss_func, cur_e, args):

    net.train()
    begin_time = datetime.datetime.now()

    train_loss = 0.0
    batch_num = int(len(data_loader.dataset) / args.batch_size)

    for i, data in enumerate(data_loader, 0):
        print('batch:%d/%d' % (i, batch_num))
        img, label = data
        img, label = img.cuda(), label.cuda()

        opt.zero_grad()

        output = net(img)[1]
        loss = loss_func(output, label)
        loss.backward()
        opt.step()

        # loss
        train_loss += loss

    end_time = datetime.datetime.now()
    delta_time = (end_time - begin_time)
    delta_seconds = (delta_time.seconds * 1000 + delta_time.microseconds) / 1000

    print('epoch:%d loss:%.4f time:%.4f' % (cur_e, train_loss.cpu(), (delta_seconds)))


def net_test(net, data_loader):
    num = len(data_loader.dataset)
    correct = 0
    net.eval()
    with t.no_grad():
        for i, data in enumerate(data_loader, 0):
            img, label = data  # cpu
            img, label = img.cuda(), label.cuda()  # gpu
            output = net(img)[1]

            predict_label = t.argmax(output, dim=1)
            correct += (predict_label == label).sum()

    return correct.cpu().numpy() * 1.0 / num


def main():
    parser = argparse.ArgumentParser(description='AID_PRETRAIN')
    parser.add_argument('--batch_size', type=int, default=128, help='training batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epoch', type=int, default=20, help='training epoch')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    args = parser.parse_args()

    data_tf = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.0], [1.0])
        ]
    )
    print('loading data')

    mnist_train_data = svhn.SVHN(data_root_path, split='train', transform=data_tf, download=True)
    mnist_test_data = svhn.SVHN(data_root_path, split='test', transform=data_tf, download=True)

    train_dataloader = DataLoader(mnist_train_data, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(mnist_test_data, batch_size=args.batch_size, shuffle=True)

    loss_func = t.nn.CrossEntropyLoss()
    net = resnet101(pretrained=False)
    net = t.nn.DataParallel(net).cuda()

    optimizer = optim.SGD(params=net.parameters(), lr=args.learning_rate, momentum=args.momentum)
    max_acc = 0.0
    print('training SVHN with resnet')
    for e in range(args.epoch):
        net_train(net, train_dataloader, optimizer, loss_func, e, args)
        test_acc = net_test(net, test_dataloader)
        print('EPOCH:%d TEST ACC:%f' % (e, test_acc))
        sys.stdout.flush()
        if test_acc > max_acc:
            max_acc = test_acc

    f = open('./result/train_svhn_on_resnet.txt', mode='w')
    f.write('max_acc:%.4f' % (max_acc))
    f.close()


if __name__ == '__main__':
    main()
