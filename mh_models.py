import torch 
import torchvision
from torchvision.transforms import Compose, Lambda, ToTensor
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST, SVHN
from torchvision.models import vgg16_bn, resnet18, vit_b_16
from torch.nn import Module, Linear, CrossEntropyLoss
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from mh_data import base_transform

def getModel(model_name, n_classes):
        if model_name.lower() == 'vgg':
            transform = torchvision.models.VGG16_BN_Weights.IMAGENET1K_V1.transforms()
            model = vgg16_bn(weights=torchvision.models.VGG16_BN_Weights.IMAGENET1K_V1)
            for param in model.parameters():
                param.requires_grad = False
            model.classifier[-1] = Linear(4096, n_classes)
        elif model_name.lower() == 'resnet':
            transform = torchvision.models.ResNet18_Weights.IMAGENET1K_V1.transforms()
            model = resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
            for param in model.parameters():
                param.requires_grad = False
            model.fc = Linear(512, n_classes)
        elif model_name.lower() == 'vit':
            transform = torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1.transforms()
            model = vit_b_16(weights=torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1)
            for param in model.parameters():
                param.requires_grad = False
            model.heads = Linear(768, n_classes)
        else:
            raise ValueError('Model not found')
        return model, transform
            
def _finetune(model_name, ds, num_classes=10, train_split=0.8, batch_size=256, patience=5, device='cuda', num_workers=4, lr=1e-3, n_epochs=10):
    print(f'Fine-tuning on {len(ds)} samples...')
    ##  Model preparation...
    model, transform = getModel(model_name, num_classes)
    model.to(device)
    model.train()
    transform = Compose([base_transform(), transform])
    ##  Data preparation...
    ds.transform = transform
    ds_train, ds_val = random_split(ds, [int(len(ds) * train_split), len(ds) - int(len(ds) * train_split)])
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=True)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=True)
    ##  Training...
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = CrossEntropyLoss()
    best_train_loss, best_val_loss = float('inf'), float('inf')
    best_model_state = model.state_dict()
    n_patience = 0
    for epoch in range(n_epochs):
        train_loss = 0
        model.train()
        for x, y in dl_train:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = model(x)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(dl_train)
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for x, y in dl_val:
                x, y = x.to(device), y.to(device)
                y_hat = model(x)
                loss = criterion(y_hat, y)
                val_loss += loss.item()
            val_loss /= len(dl_val)
            if val_loss < best_val_loss:
                best_train_loss = train_loss
                best_val_loss = val_loss
                best_model_state = model.state_dict()
                n_patience = 0
            else:
                n_patience += 1
                if n_patience == patience:
                    print(f'Early stopping at epoch {epoch + 1}')
                    break
        model.load_state_dict(best_model_state)
        print(f'Epoch: {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Best Val Loss: {best_val_loss:.4f}')
    return model, transform, best_train_loss, best_val_loss

def finetune(model_name, ds, num_classes=10, train_split=0.8, batch_size=256, patience=5, device='cuda', num_workers=4, lr=1e-3, n_epochs=10, n_tries=3, loss_thresh=0.5):
    cnt = 0
    best_val_loss = float('inf')
    while best_val_loss > loss_thresh and cnt < n_tries:
        print(f'Try {cnt + 1}...')
        model, transform, train_loss, val_loss = _finetune(model_name, ds, num_classes, train_split, batch_size, patience, device, num_workers, lr, n_epochs)
        cnt -=- 1
        if val_loss < best_val_loss:
            best_model = model
            best_transform = transform
            best_train_loss = train_loss
            best_val_loss = val_loss
            print(f'Best model found so far with validation loss {best_val_loss}')
    return best_model, best_transform, best_train_loss, best_val_loss


if __name__ == '__main__':
    import mh_data as mhd
    ds_unlabelled, ds_fine_tuning, ds_optimization = mhd.dataSplit('mnist')
    model, transform, train_loss, val_loss = finetune('resnet', ds_fine_tuning)
    print(train_loss, val_loss)

    