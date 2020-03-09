import torchvision
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
