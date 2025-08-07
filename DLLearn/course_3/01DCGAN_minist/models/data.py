from torch.utils.data import DataLoader
from torchvision import utils, datasets, transforms

class ReadData():
    def __init__(self,dataroot,image_size=64):
        self.root=dataroot
        self.image_size=image_size
        self.dataset=self.getdataset()
    def getdataset(self):
        #3.dataset
        train_data = datasets.MNIST(
            root=self.root,
            train=True,
            transform=transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ]),
            download=True
        )
        test_data = datasets.MNIST(
            root=self.root,
            train=False,
            transform=transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        )
        dataset = train_data+test_data
        # print(f'Total Size of Dataset: {len(dataset)}')
        return dataset

    def getdataloader(self,batch_size=128):
        #4.dataloader
        dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
        return dataloader
