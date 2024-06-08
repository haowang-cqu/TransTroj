from torchvision import datasets, transforms
from torch.utils.data import random_split
import numpy as np
import os


transform = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR, antialias="warn"),
    transforms.CenterCrop(224)
])


def prepare_cifar10(dir):
    cifar10_train = datasets.CIFAR10(
        root='.cache',
        train=True,
        download=True,
        transform=transform
    )
    cifar10_test = datasets.CIFAR10(
        root='.cache',
        train=False,
        download=True,
        transform=transform
    )
    cifar10_train_images = np.array([np.array(image) for image, _ in cifar10_train])
    cifar10_train_labels = np.array([[label] for _, label in cifar10_train])
    cifar10_test_images = np.array([np.array(image) for image, _ in cifar10_test])
    cifar10_test_labels = np.array([[label] for _, label in cifar10_test])

    np.savez(os.path.join(dir, "train.npz"), x=cifar10_train_images, y=cifar10_train_labels)
    np.savez(os.path.join(dir, "test.npz"), x=cifar10_test_images, y=cifar10_test_labels)


def prepare_cifar100(dir):
    cifar100_train = datasets.CIFAR100(
        root='.cache',
        train=True,
        download=True,
        transform=transform
    )
    cifar100_test = datasets.CIFAR100(
        root='.cache',
        train=False,
        download=True,
        transform=transform
    )
    cifar100_train_images = np.array([np.array(image) for image, _ in cifar100_train])
    cifar100_train_labels = np.array([[label] for _, label in cifar100_train])
    cifar100_test_images = np.array([np.array(image) for image, _ in cifar100_test])
    cifar100_test_labels = np.array([[label] for _, label in cifar100_test])

    np.savez(os.path.join(dir, "train.npz"), x=cifar100_train_images, y=cifar100_train_labels)
    np.savez(os.path.join(dir, "test.npz"), x=cifar100_test_images, y=cifar100_test_labels)


def prepare_gtsrb(dir):
    gtsrb_train = datasets.GTSRB(
        root='.cache',
        split='train',
        download=True,
        transform=transform
    )
    gtsrb_test = datasets.GTSRB(
        root='.cache',
        split='test',
        download=True,
        transform=transform
    )
    gtsrb_train_images = np.array([np.array(image) for image, _ in gtsrb_train])
    gtsrb_train_labels = np.array([[label] for _, label in gtsrb_train])
    gtsrb_test_images = np.array([np.array(image) for image, _ in gtsrb_test])
    gtsrb_test_labels = np.array([[label] for _, label in gtsrb_test])

    np.savez(os.path.join(dir, "train.npz"), x=gtsrb_train_images, y=gtsrb_train_labels)
    np.savez(os.path.join(dir, "test.npz"), x=gtsrb_test_images, y=gtsrb_test_labels)


def prepare_pet(dir):
    pet_train = datasets.OxfordIIITPet(
        root='.cache',
        split='trainval',
        target_types="category",
        download=True,
        transform=transform
    )
    pet_test = datasets.OxfordIIITPet(
        root='.cache',
        split='test',
        target_types="category",
        download=True,
        transform=transforms.transform
    )
    pet_train_images = np.array([np.array(image) for image, _ in pet_train])
    pet_train_labels = np.array([[label] for _, label in pet_train])
    pet_test_images = np.array([np.array(image) for image, _ in pet_test])
    pet_test_labels = np.array([[label] for _, label in pet_test])

    np.savez(os.path.join(dir, "train.npz"), x=pet_train_images, y=pet_train_labels)
    np.savez(os.path.join(dir, "test.npz"), x=pet_test_images, y=pet_test_labels)


def prepare_caltech101(dir):
    caltech101 = datasets.Caltech101(
        root='.cache',
        target_type='category',
        download=True,
        transform=transform
    )
    # random split
    caltech101_train, caltech101_test = random_split(caltech101, [int(0.8 * len(caltech101)), len(caltech101) - int(0.8 * len(caltech101))])
    caltech101_train_images = np.array([np.array(image) for image, _ in caltech101_train])
    caltech101_train_labels = np.array([[label] for _, label in caltech101_train])
    caltech101_test_images = np.array([np.array(image) for image, _ in caltech101_test])
    caltech101_test_labels = np.array([[label] for _, label in caltech101_test])

    np.savez(os.path.join(dir, "train.npz"), x=caltech101_train_images, y=caltech101_train_labels)
    np.savez(os.path.join(dir, "test.npz"), x=caltech101_test_images, y=caltech101_test_labels)


def prepare_caltech256(dir):
    caltech256 = datasets.Caltech256(
        root='.cache',
        download=True,
        transform=transform
    )
    # random split
    caltech256_train, caltech256_test = random_split(caltech256, [int(0.8 * len(caltech256)), len(caltech256) - int(0.8 * len(caltech256))])
    caltech256_train_images = np.array([np.array(image) for image, _ in caltech256_train])
    caltech256_train_labels = np.array([[label] for _, label in caltech256_train])
    caltech256_test_images = np.array([np.array(image) for image, _ in caltech256_test])
    caltech256_test_labels = np.array([[label] for _, label in caltech256_test])

    np.savez(os.path.join(dir, "train.npz"), x=caltech256_train_images, y=caltech256_train_labels)
    np.savez(os.path.join(dir, "test.npz"), x=caltech256_test_images, y=caltech256_test_labels)


def prepare_shadow_images(dir):
    """ It is required to download ImageNet 2012 dataset from here and place the files ILSVRC2012_devkit_t12.tar.gz 
    and ILSVRC2012_img_train.tar or ILSVRC2012_img_val.tar based on split in the root directory.
    """
    imagenet = datasets.ImageNet(
        root='.cache',
        split='train',
        transform=transform
    )
    # random select 50000 images
    indices = np.random.choice(len(imagenet), 50000, replace=False)
    shadow_images = np.array([np.array(imagenet[idx][0]) for idx in indices])
    np.save(os.path.join(dir, "50000.npy"), shadow_images)


def main(root_dir):
    cifar10_dir = os.path.join(root_dir, "cifar10")
    os.makedirs(cifar10_dir, exist_ok=True)
    prepare_cifar10(cifar10_dir)

    cifar100_dir = os.path.join(root_dir, "cifar100")
    os.makedirs(cifar100_dir, exist_ok=True)
    prepare_cifar100(cifar100_dir)

    gtsrb_dir = os.path.join(root_dir, "gtsrb")
    os.makedirs(gtsrb_dir, exist_ok=True)
    prepare_gtsrb(gtsrb_dir)

    pet_dir = os.path.join(root_dir, "pet")
    os.makedirs(pet_dir, exist_ok=True)
    prepare_pet(pet_dir)

    caltech101_dir = os.path.join(root_dir, "caltech101")
    os.makedirs(caltech101_dir, exist_ok=True)
    prepare_caltech101(caltech101_dir)

    caltech256_dir = os.path.join(root_dir, "caltech256")
    os.makedirs(caltech256_dir, exist_ok=True)
    prepare_caltech256(caltech256_dir)

    shadow_images_dir = os.path.join(root_dir, "shadow_images")
    os.makedirs(shadow_images_dir, exist_ok=True)
    prepare_shadow_images(shadow_images_dir)

    
if __name__ == '__main__':
    main(root_dir='data')
