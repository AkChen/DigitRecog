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

svhn_train_data = svhn.SVHN(root_path+'SVHN',split='train',transform=data_tf,download=True)
svhn_test_data = svhn.SVHN(root_path+'SVHN',split='test',transform=data_tf,download=True)

mnist_cls_map = dict() #
svhn_cls_map = dict() #
for i in range(10):
    mnist_cls_map[i] = []
    svhn_cls_map[i] = []

# train
mnist_count = 0
svhn_count = 0

# mnist
for i,c in enumerate(mnist_train_data.targets):
    mnist_cls_map[int(c.numpy())].append(i)
    mnist_count += 1
# svhn
for i,c in enumerate(svhn_train_data.labels):
    svhn_cls_map[c].append(i)
    svhn_count += 1

# common_idx
mnist_common_idx = []
svhn_common_idx = []

for i in range(10):
    min_len = min(len(mnist_cls_map[i]),len(svhn_cls_map[i]))
    mnist_common_idx.extend(mnist_cls_map[i][:min_len])
    svhn_common_idx.extend(svhn_cls_map[i][:min_len])


# get data

train_data_dict = dict()
train_data_dict['mnist'] = np.asarray(np.asarray(mnist_train_data.data)[mnist_common_idx])
train_data_dict['svhn'] = np.asarray(np.asarray(svhn_train_data.data)[svhn_common_idx])
train_data_dict['label'] = np.asarray(mnist_train_data.targets)[mnist_common_idx]

# now test data
mnist_cls_map = dict() #
svhn_cls_map = dict() #
for i in range(10):
    mnist_cls_map[i] = []
    svhn_cls_map[i] = []

# train
mnist_count = 0
svhn_count = 0

# mnist
for i,c in enumerate(mnist_test_data.targets):
    mnist_cls_map[int(c.numpy())].append(i)
    mnist_count += 1
# svhn
for i,c in enumerate(svhn_test_data.labels):
    svhn_cls_map[c].append(i)
    svhn_count += 1

# common_idx
mnist_common_idx = []
svhn_common_idx = []

for i in range(10):
    min_len = min(len(mnist_cls_map[i]),len(svhn_cls_map[i]))
    mnist_common_idx.extend(mnist_cls_map[i][:min_len])
    svhn_common_idx.extend(svhn_cls_map[i][:min_len])


# get data

test_data_dict = dict()
test_data_dict['mnist'] = np.asarray(np.asarray(mnist_test_data.data)[mnist_common_idx])
test_data_dict['svhn'] = np.asarray(np.asarray(svhn_test_data.data)[svhn_common_idx])
test_data_dict['label'] = np.asarray(mnist_test_data.targets)[mnist_common_idx]



np.save('./biomodal/biomodal_train',train_data_dict)
np.save('./biomodal/biomodal_test',test_data_dict)