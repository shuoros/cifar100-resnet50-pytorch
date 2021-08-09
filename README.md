# CIFAR100 ResNet50 transfer learning in Pytorch
- Computational Intelligence course final project.

- Instructed by [Ali Tourani](https://scholar.google.com/citations?user=_VkNRkUAAAAJ&hl=en "‪Ali Tourani‬ - ‪Google Scholar‬‬") at [University of Guilan](https://guilan.ac.ir/en/home "University of Guilan‬").

## Dataset
I used CIFAR-100 as dataset and you can read the description below according to the [docs](https://www.cs.toronto.edu/~kriz/cifar.html).
>The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. This dataset is just like the CIFAR-10, except it has 100 classes containing 600 images each.

### Data Augmentation and Preparation
For data augmentation, I used `torchvision.transforms` and feed them to `torchvision.datasets.CIFAR100` to apply them on dataset.
```python
data_transforms = {
    'train':
    transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]),
    'test':
    transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]),
}

data_sets = {
    'train': torchvision.datasets.CIFAR100(root='data', train=True, download=True, transform=data_transforms['train']),
    'test': torchvision.datasets.CIFAR100(root='data', download=True, transform=data_transforms['test'])
}
```

Finally, To prepare the data for training and to divide it into batch sizes, I fed dataset to the `torch.utils.data.DataLoader`.
```python
dataloaders = {
    'train':
    torch.utils.data.DataLoader(data_sets['train'],
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=0),
    'test':
    torch.utils.data.DataLoader(data_sets['test'],
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=0)
}
```
## Model
The model is a pretrained ResNet50 with a .5 Dropout and 6 Linear layers that each one has a .2 Dropout as fc (fully connected layer) for top of the model.

I used `CrossEntropyLoss()` for criterion and `SGD` optimizer for optimizition.
```python
model = models.resnet50(pretrained=True)

model = model.cuda() if use_cuda else model
    
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 1024)
model.fc = nn.Sequential(
    torch.nn.Dropout(0.5),
    torch.nn.Linear(num_ftrs, 1024),
    torch.nn.Dropout(0.2),
    torch.nn.Linear(1024, 512),
    torch.nn.Dropout(0.2),
    torch.nn.Linear(512, 256),
    torch.nn.Dropout(0.2),
    torch.nn.Linear(256, 128),
    torch.nn.Dropout(0.2),
    torch.nn.Linear(128, len(data_sets['train'].classes))
)

model.fc = model.fc.cuda() if use_cuda else model.fc

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
```
A `torch.optim.lr_scheduler.StepLR` was used to decays the learning rate of each parameter group by gamma every step_size epochs [see docs here](https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.StepLR) 


Example from docs:
```python
# Assuming optimizer uses lr = 0.05 for all groups
# lr = 0.05     if epoch < 30
# lr = 0.005    if 30 <= epoch < 60
# lr = 0.0005   if 60 <= epoch < 90
# ...
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
for epoch in range(100):
    train(...)
    validate(...)
    scheduler.step()
```
## Results
After
- *20* epochs of train
- With batch size of *64*

and

- Learning rate of **3e-3**

The following results were obtained:

![](https://github.com/shuoros/cifar100-resnet50-pytorch/blob/main/etc/acc.png)

![](https://github.com/shuoros/cifar100-resnet50-pytorch/blob/main/etc/loss.png)

## test

For the final test, 12 images were randomly selected from the test set and given to the model to classify them.

The prediction of model is at the top of each image.

If that prediction is true the word "True" would be printed next to it and if it is not true the word "False" would be printed next to it.

![](https://github.com/shuoros/cifar100-resnet50-pytorch/blob/main/etc/raccoon.png)
![](https://github.com/shuoros/cifar100-resnet50-pytorch/blob/main/etc/bottle.png)
![](https://github.com/shuoros/cifar100-resnet50-pytorch/blob/main/etc/procupine.png)
![](https://github.com/shuoros/cifar100-resnet50-pytorch/blob/main/etc/lizard.png)
![](https://github.com/shuoros/cifar100-resnet50-pytorch/blob/main/etc/leopard.png)
![](https://github.com/shuoros/cifar100-resnet50-pytorch/blob/main/etc/willow_tree.png)
![](https://github.com/shuoros/cifar100-resnet50-pytorch/blob/main/etc/wardrobe.png)
![](https://github.com/shuoros/cifar100-resnet50-pytorch/blob/main/etc/poppy.png)
![](https://github.com/shuoros/cifar100-resnet50-pytorch/blob/main/etc/poppy%20(2).png)
![](https://github.com/shuoros/cifar100-resnet50-pytorch/blob/main/etc/chimpanzee.png)
![](https://github.com/shuoros/cifar100-resnet50-pytorch/blob/main/etc/clock.png)
![](https://github.com/shuoros/cifar100-resnet50-pytorch/blob/main/etc/castle.png)
![](https://github.com/shuoros/cifar100-resnet50-pytorch/blob/main/etc/table.png)
![](https://github.com/shuoros/cifar100-resnet50-pytorch/blob/main/etc/palm_tree.png)
![](https://github.com/shuoros/cifar100-resnet50-pytorch/blob/main/etc/pine_tree.png)
![](https://github.com/shuoros/cifar100-resnet50-pytorch/blob/main/etc/snale.png)
![](https://github.com/shuoros/cifar100-resnet50-pytorch/blob/main/etc/seal.png)
![](https://github.com/shuoros/cifar100-resnet50-pytorch/blob/main/etc/rose.png)
![](https://github.com/shuoros/cifar100-resnet50-pytorch/blob/main/etc/can.png)
![](https://github.com/shuoros/cifar100-resnet50-pytorch/blob/main/etc/tractor.png)

## Download
You can downlad weights of model in a h5 file with accuracy of 98.65% in testset and 95.4% in trainset from [here](https://drive.google.com/file/d/144gXaJr6VG9WuemC2q4x8c1jizKJcUyI/view?usp=sharing)
