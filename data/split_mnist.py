import torchvision
import numpy as np
from torchvision.datasets import mnist,svhn

data_tf = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.0],[1.0])
    ]
)

root_path = './'


mnist_train_data = mnist.MNIST(root_path,train=True,transform=data_tf,download=True)
mnist_test_data = mnist.MNIST(root_path,train=False,transform=data_tf,download=True)

# split train set
mnist_cls_map = dict() #
for i in range(10):
    mnist_cls_map[i] = []
for i,c in enumerate(mnist_train_data.targets):
    mnist_cls_map[int(c.numpy())].append(i)

mnist_a_idx = []
mnist_b_idx = []
for i in range(10):
    num = len(mnist_cls_map[i])
    if num % 2 != 0:
        num -= 1
    #
    half_num = int(num/2)
    mnist_a_idx.extend(mnist_cls_map[i][:half_num])
    mnist_b_idx.extend(mnist_cls_map[i][half_num:half_num+half_num])

train_data_a = mnist_train_data.data[mnist_a_idx]
train_data_b = mnist_train_data.data[mnist_b_idx]
train_label = mnist_train_data.targets[mnist_a_idx]



# test
mnist_cls_map = dict() #
for i in range(10):
    mnist_cls_map[i] = []
for i,c in enumerate(mnist_test_data.targets):
    mnist_cls_map[int(c.numpy())].append(i)

mnist_a_idx = []
mnist_b_idx = []
for i in range(10):
    num = len(mnist_cls_map[i])
    if num % 2 != 0:
        num -= 1
    #
    half_num = int(num/2)
    mnist_a_idx.extend(mnist_cls_map[i][:half_num])
    mnist_b_idx.extend(mnist_cls_map[i][half_num:half_num+half_num])

test_data_a = mnist_test_data.data[mnist_a_idx]
test_data_b = mnist_test_data.data[mnist_b_idx]
test_label = mnist_test_data.targets[mnist_a_idx]


train_data_dict = dict()

train_data_dict['label'] = train_label
#t = [[d]for d in train_data_a]


train_data_dict['a'] = np.asarray([[np.asarray(d),np.asarray(d),np.asarray(d)]for d in train_data_a])
train_data_dict['b'] = np.asarray([[np.asarray(d),np.asarray(d),np.asarray(d)]for d in train_data_b])
np.save('./splitmodal/split_mnist_train',train_data_dict)

test_data_dict = dict()

test_data_dict['label'] = test_label
test_data_dict['a'] = np.asarray([[np.asarray(d),np.asarray(d),np.asarray(d)]for d in test_data_a])
test_data_dict['b'] = np.asarray([[np.asarray(d),np.asarray(d),np.asarray(d)]for d in test_data_b])
np.save('./splitmodal/split_mnist_test',test_data_dict)