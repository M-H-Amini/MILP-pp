import torch 
import torchvision
from torchvision.transforms import Compose, Lambda, ToTensor
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST, SVHN
from torchvision.models import vgg16_bn, resnet18, vit_b_16
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans
import numpy as np
from mh_custom_dataset import MHSyntheticDataset

def dataSplit(ds_name, h_init=0.05, s=500):
    transform = Compose([
        base_transform(),
        torchvision.models.VGG16_BN_Weights.IMAGENET1K_V1.transforms()
    ])
    
    if ds_name == 'mnist':
        data = MNIST(root='./data', train=True, download=True, transform=transform)
    elif ds_name == 'cifar10':
        data = CIFAR10(root='./data', train=True, download=True, transform=transform)
    elif ds_name == 'fashionmnist':
        data = FashionMNIST(root='./data', train=True, download=True, transform=transform)
    elif ds_name == 'svhn':
        data = SVHN(root='./data', split='train', download=True, transform=transform)
    elif ds_name == 'hu_cifar10':
        data = MHSyntheticDataset(root='./Hu_Cifar10', transform=transform)
    elif ds_name == 'hu_imagenet':
        data = MHSyntheticDataset(root='./Hu_Imagenet', transform=transform)


    # data = random_split(data, [15000, len(data) - 15000])[0]  ##  For debugging...
    print('Total number of samples:', len(data))
    n_labelled = int(len(data) * h_init)
    if n_labelled < 1000:
        s = n_labelled // 2
    print('Number of labelled samples:', n_labelled)
    indices = sample(data, n_labelled)
    unlabelled_indices = list(set(range(len(data))) - set(indices))
    optimization_indices = np.random.choice(indices, s, replace=False)
    fine_tuning_indices = list(set(indices) - set(optimization_indices))
    ds_unlabelled = Subset(data, unlabelled_indices)
    ds_fine_tuning = Subset(data, fine_tuning_indices)
    ds_optimization = Subset(data, optimization_indices)
    print('Number of unlabelled samples:', len(ds_unlabelled))
    print('Number of fine tuning samples:', len(ds_fine_tuning))
    print('Number of optimization samples:', len(ds_optimization))
    return ds_unlabelled, ds_fine_tuning, ds_optimization



def sample(dataset, n, device="cuda", num_workers=4):
    model_vgg = vgg16_bn(weights=torchvision.models.VGG16_BN_Weights.IMAGENET1K_V1)
    model_vgg.classifier = torch.nn.Sequential(*list(model_vgg.classifier.children())[:-3])
    model_vgg.to(device).half().eval()
    dl = DataLoader(dataset, batch_size=2048, shuffle=False, num_workers=num_workers)
    ##  Feature extraction...
    X = []
    for x, y in tqdm(dl):
        x = x.to(device).half()
        with torch.no_grad():
            x = model_vgg(x)
        X.append(x)
    X = torch.cat(X).cpu().numpy()
    print('Feature shape:', X.shape)
    ##  PCA...
    pca = PCA(n_components=2)
    X = pca.fit_transform(X)
    ##  Clustering...
    n_clusters = 50
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X)
    ##  Randomly sample from each cluster to get n indices...
    indices = []
    for i in range(n_clusters):
        idx = np.where(kmeans.labels_ == i)[0].tolist()
        if len(idx) >= n // n_clusters:  ##  If the cluster has more samples than required...
            idx = np.random.choice(idx, n // n_clusters, replace=False).tolist()
            indices.extend(idx)
    while len(set(indices)) < n:
        idx = np.random.choice(len(X), 1, replace=False)[0]
        if not idx in indices:
            indices.append(idx)
    print('No of sampled indices:', len(indices))
    return indices
    
def base_transform():
    return Compose([
        ToTensor(),
        Lambda(lambda x: torch.cat([x, x, x], dim=0) if x.shape[0] == 1 else x),
    ])
        

if __name__ == '__main__':
    # ds_unlabelled, ds_fine_tuning, ds_optimization = dataSplit('svhn')
    # ds_unlabelled, ds_fine_tuning, ds_optimization = dataSplit('hu_cifar10')
    ds_unlabelled, ds_fine_tuning, ds_optimization = dataSplit('hu_imagenet')
    