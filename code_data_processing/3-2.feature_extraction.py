import torch
from torch import nn
from torchvision import models as torch_models
import torchvision.transforms as transforms
import numpy as np
import os
from PIL import Image
from skimage import io
from tqdm import tqdm
import pandas as pd
import argparse


class ResNet_extractor(nn.Module):
    def __init__(self, layers=101):
        super().__init__()
        if layers == 18:
            self.resnet = torch_models.resnet18(pretrained=True)
        elif layers == 34:
            self.resnet = torch_models.resnet34(pretrained=True)
        elif layers == 50:
            self.resnet = torch_models.resnet50(pretrained=True)
        elif layers == 101:
            self.resnet = torch_models.resnet101(pretrained=True)
        else:
            raise(ValueError('Layers must be 18, 34, 50 or 101.'))

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        return x


def main():
    parser = argparse.ArgumentParser(description='The start and end positions in the file list')
    parser.add_argument('--gpu', type=str, default='0', help='GPU device')
    parser.add_argument('--start', type=float, default=0.0, help='start position')
    parser.add_argument('--end', type=float, default=0.01, help='end position')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    dataset = 'TCGA-BRCA'
    data_dir = '../data/{:s}'.format(dataset)
    all_patch_indices = np.load('{:s}/20x_512x512/clustering_results/all_patch_indices_refined.npy'.format(data_dir),
                                allow_pickle=True).item()

    img_dir = '{:s}/20x_normalized'.format(data_dir)
    save_dir = '{:s}/20x_features_resnet101'.format(data_dir)
    os.makedirs(save_dir, exist_ok=True)

    data_transform = transforms.Compose([transforms.Resize(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    batch_size = 64
    model = ResNet_extractor(layers=101).cuda()
    model = model.eval()

    slides_list = pd.read_csv('{:s}/slide_selection_final.txt'.format(data_dir), header=None)
    slides_list = list(slides_list[0].values)
    N = len(slides_list)
    slides_to_be_processed = slides_list[int(N * args.start):int(N * args.end)]

    count = 0
    for slide_name in tqdm(slides_to_be_processed):
        indices = all_patch_indices[slide_name]
        count += 1
        # if count < 49:
        #     continue

        tumor_indices = indices['tumor']
        N_tumor_patch = len(tumor_indices)
        feature_list = []
        index_list = []
        for batch_idx in range(0, N_tumor_patch, batch_size):
            end = batch_idx + batch_size if batch_idx+batch_size < N_tumor_patch else N_tumor_patch
            indices = tumor_indices[batch_idx: end]
            images = []
            for idx in indices:
                image = Image.open('{:s}/{:s}/{:d}.png'.format(img_dir, slide_name, int(idx))).convert('RGB')
                image_tensor = data_transform(image).unsqueeze(0)
                images.append(image_tensor)
            images = torch.cat(images, dim=0)

            features = model(images.cuda())
            feature_list.append(features.detach().cpu())
            index_list += list(indices)
            del features

        feature_list = torch.cat(feature_list, dim=0)
        np.save('{:s}/{:s}.npy'.format(save_dir, slide_name), {'features': feature_list.numpy(), 'indices': np.array(index_list)})


if __name__ == '__main__':
    main()