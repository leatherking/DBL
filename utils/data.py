import numpy as np
from torchvision import datasets, transforms
from utils.toolkit import split_images_labels #
import os
from PIL import Image
import pdb
import shutil

class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None


class iCIFAR10(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = []
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
        ),
    ]

    class_order = np.arange(10).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR10("/homedata/myc/myc_ssd/DBL/datasets", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR10("/homedata/myc/myc_ssd/DBL/datasets", train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )


class iCIFAR100(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = []
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)
        ),
    ]

    class_order = np.arange(100).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100("/homedata/myc/myc_ssd/DBL/datasets", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100("/homedata/myc/myc_ssd/DBL/datasets", train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )


class iImageNet1000(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        assert 0, "You should specify the folder of your dataset"
        train_dir = "[DATA-PATH]/train/"
        test_dir = "[DATA-PATH]/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iImageNet100(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(100).tolist()

    def download_data(self):
        train_dir = "/home/myc/CNCE/datasets/seed_1993_subset_100_imagenet/data/train/"
        test_dir = "/home/myc/CNCE/datasets/seed_1993_subset_100_imagenet/data/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iTinyImageNet(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = []
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]

    class_order = np.arange(200).tolist()

    def download_data(self):
        root = '/homedata/myc/myc_ssd/DBL/datasets/tinyimagenet'
        if os.path.isdir(root) and len(os.listdir(root)) > 0:
            print('Download not needed, files already on disk.')
        else:
            raise ValueError("None")
            print('Downloading dataset')
            gdd.download_file_from_google_drive(
                    file_id='1Sy3ScMBr0F4se8VZ6TAwDYF-nNGAAdxj',
                    dest_path=os.path.join(root, 'tiny-imagenet-processed.zip'),
                    unzip=True)
        self.train_data = []
        self.train_targets = []
        
        for num in range(20):
            self.train_data.append(np.load(os.path.join(
                root, 'processed/x_%s_%02d.npy' %
                    ('train', num+1)))) # if self.train else 'val'
            self.train_targets.append(np.load(os.path.join(
                root, 'processed/y_%s_%02d.npy' % (
                    'train', num+1
                )
            )))
        self.train_targets = np.concatenate(np.array(self.train_targets))
        self.train_data = np.concatenate(np.array(self.train_data))
        self.train_data = np.uint8(255 * self.train_data)

        self.test_data = []
        self.test_targets = []
        for num in range(20):
            self.test_data.append(np.load(os.path.join(
                root, 'processed/x_%s_%02d.npy' %
                    ('val', num+1))))
            self.test_targets.append(np.load(os.path.join(
                root, 'processed/y_%s_%02d.npy' % (
                    'val', num+1
                )
            )))
        self.test_targets = np.concatenate(np.array(self.test_targets))
        self.test_data = np.concatenate(np.array(self.test_data))
        self.test_data = np.uint8(255 * self.test_data)

class iCUB200(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(200).tolist()

    def download_data(self):
        train_dir = "/home/myc/PyCIL/datasets/CUB_200_2011/train/"
        test_dir = "/home/myc/PyCIL/datasets/CUB_200_2011/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

def split_CUB(root):
    train_dir = os.path.join(root, 'train')
    test_dir = os.path.join(root, 'val')
    img_dir = os.path.join(root,'images')
    images_path = {}

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    
    for f in os.listdir(img_dir):
        if not os.path.exists(os.path.join(train_dir,f)):
            os.makedirs(os.path.join(train_dir,f))
        if not os.path.exists(os.path.join(test_dir,f)):
            os.makedirs(os.path.join(test_dir,f))
    
    with open(os.path.join(root, 'images.txt')) as f:
        for line in f:
            image_id, path = line.split()
            images_path[image_id] = path
    
    count_test = 0
    with open(os.path.join(root, 'train_test_split.txt')) as f:
        for line in f:
            image_id, is_train = line.split()
            if int(is_train):
                shutil.copyfile(os.path.join(img_dir, images_path[image_id]), os.path.join(train_dir, images_path[image_id]))
            else:
                shutil.copyfile(os.path.join(img_dir, images_path[image_id]), os.path.join(test_dir, images_path[image_id]))
            count_test += 1
            print(count_test)

if __name__ == '__main__':
    # idata = iTinyImageNet()
    # idata.download_data()
    # split_CUB('datasets/CUB_200_2011')
    idata = iCUB200()
    idata.download_data()