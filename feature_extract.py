# import torch
# import torchvision.transforms as T
# from torchvision.models import resnet50, vit_b_16, vit_b_32, vit_l_16, vit_l_32
# from transformers import CLIPProcessor, CLIPModel
# from PIL import Image
# from tqdm import tqdm
# import os
# import numpy as np
# import deepcore.nets as nets
# import deepcore.datasets as datasets
# import deepcore.methods as methods

# # Set paths
# images_dir = "path_to_images"  # Replace with your image directory path
# output_dir = "/n/fs/dk-diffusion/repos/DeepCore/features/waterbirds/"  # Directory to save features
# os.makedirs(output_dir, exist_ok=True)

# channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, dst_val= datasets.waterbirds('/n/fs/visualai-scr/Data/waterbird/waterbird_complete95_forest2water2')
# train_loader = torch.utils.data.DataLoader(dst_train, shuffle=False, batch_size=512)

# # Transformation for the input images
# transform = T.Compose([
#     T.Resize((224, 224)),
#     T.ToTensor(),
#     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

# # Define feature extraction function
# def save_features(model, model_name, device="cuda"):
#     model = model.to(device).eval()
#     features = []
#     with torch.no_grad():
#         for inputs, labels, c_labels, _ in tqdm(train_loader):
#             image = inputs.to(device)
#             feature = model(image)#.cpu().squeeze(0)
#             features.append(feature)
#         # features=np.array(features, dtype=object)
#         features=torch.cat(features).cpu().numpy()
#     torch.save(features, os.path.join(output_dir, f"{model_name}_features.pt"))

# # Device setup
# device = "cuda" if torch.cuda.is_available() else "cpu"

# # # **Model 1**: RN50 - Imagenet1k
# # rn50_1k = resnet50(pretrained=True)
# # rn50_1k = torch.nn.Sequential(*list(rn50_1k.children())[:-1])  # Remove the classifier
# # save_features(rn50_1k,  "RN50_Imagenet1k", device)

# # **Model 2**: ViT - Imagenet1k
# vit_1k = vit_b_16(pretrained=True)
# vit_1k.heads = torch.nn.Identity()  # Remove the classification head
# save_features(vit_1k, "ViT_Imagenet1k", device)

# # **Model 3**: ViT - Imagenet1k
# vit_1k = vit_b_32(pretrained=True)
# vit_1k.heads = torch.nn.Identity()  # Remove the classification head
# save_features(vit_1k, "ViTb32_Imagenet1k", device)

# # vit_1k = vit_l_16(pretrained=True)
# # vit_1k.heads = torch.nn.Identity()  # Remove the classification head
# # save_features(vit_1k, "ViTl16_Imagenet1k", device)

# # vit_1k = vit_l_32(pretrained=True)
# # vit_1k.heads = torch.nn.Identity()  # Remove the classification head
# # save_features(vit_1k, "ViTl32_Imagenet1k", device)

# dinov2_vitb14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
# dinov2_vitb14.heads = torch.nn.Identity()  # Remove the classification head
# save_features(dinov2_vitb14, "dinov2_vitb14", device)


# import os
# import clip
# import torch
# import numpy as np
# from sklearn.linear_model import LogisticRegression
# from torch.utils.data import DataLoader
# from torchvision.datasets import CIFAR100
# from tqdm import tqdm

# # Load the model
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load('ViT-B/32', device)

# def get_features(dataset):
#     all_features = []
#     all_labels = []
    
#     with torch.no_grad():
#         for images, labels, _, _ in tqdm(DataLoader(dataset, batch_size=100)):
#             features = model.encode_image(images.to(device))

#             all_features.append(features)
#             all_labels.append(labels)

#     return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()

# # Calculate the image features
# train_features, train_labels = get_features(dst_train)
# test_features, test_labels = get_features(dst_test)
# model_name= 'CLIP-ViTB32'
# torch.save(train_features, os.path.join(output_dir, f"{model_name}_features.pt"))








import torch
import torchvision.transforms as T
from torchvision.models import resnet50, vit_b_16, vit_b_32, vit_l_16, vit_l_32
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from tqdm import tqdm
import os
import numpy as np
import deepcore.nets as nets
import deepcore.datasets as datasets
import deepcore.methods as methods

# Set paths
images_dir = "path_to_images"  # Replace with your image directory path
output_dir = "/n/fs/dk-diffusion/repos/DeepCore/features/metashift/"  # Directory to save features
os.makedirs(output_dir, exist_ok=True)

channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, dst_val= datasets.Metashift('/n/fs/visualai-scr/Data/')
train_loader = torch.utils.data.DataLoader(dst_train, shuffle=False, batch_size=512)

# Transformation for the input images
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define feature extraction function
def save_features(model, model_name, device="cuda"):
    model = model.to(device).eval()
    features = []
    with torch.no_grad():
        for inputs, labels, c_labels, _ in tqdm(train_loader):
            image = inputs.to(device)
            feature = model(image)#.cpu().squeeze(0)
            features.append(feature)
        # features=np.array(features, dtype=object)
        features=torch.cat(features).cpu().numpy()
    torch.save(features, os.path.join(output_dir, f"{model_name}_features.pt"))

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# **Model 1**: RN50 - Imagenet1k
rn50_1k = resnet50(pretrained=True)
rn50_1k = torch.nn.Sequential(*list(rn50_1k.children())[:-1])  # Remove the classifier
save_features(rn50_1k,  "RN50_Imagenet1k", device)

# # **Model 2**: ViT - Imagenet1k
# vit_1k = vit_b_16(pretrained=True)
# vit_1k.heads = torch.nn.Identity()  # Remove the classification head
# save_features(vit_1k, "ViT_Imagenet1k", device)

# # **Model 3**: ViT - Imagenet1k
# vit_1k = vit_b_32(pretrained=True)
# vit_1k.heads = torch.nn.Identity()  # Remove the classification head
# save_features(vit_1k, "ViTb32_Imagenet1k", device)

# # vit_1k = vit_l_16(pretrained=True)
# # vit_1k.heads = torch.nn.Identity()  # Remove the classification head
# # save_features(vit_1k, "ViTl16_Imagenet1k", device)

# # vit_1k = vit_l_32(pretrained=True)
# # vit_1k.heads = torch.nn.Identity()  # Remove the classification head
# # save_features(vit_1k, "ViTl32_Imagenet1k", device)

# dinov2_vitb14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
# dinov2_vitb14.heads = torch.nn.Identity()  # Remove the classification head
# save_features(dinov2_vitb14, "dinov2_vitb14", device)


# import os
# import clip
# import torch
# import numpy as np
# from sklearn.linear_model import LogisticRegression
# from torch.utils.data import DataLoader
# from torchvision.datasets import CIFAR100
# from tqdm import tqdm

# # Load the model
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load('ViT-B/32', device)

# def get_features(dataset):
#     all_features = []
#     all_labels = []
    
#     with torch.no_grad():
#         for images, labels, _, _ in tqdm(DataLoader(dataset, batch_size=100)):
#             features = model.encode_image(images.to(device))

#             all_features.append(features)
#             all_labels.append(labels)

#     return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()

# # Calculate the image features
# train_features, train_labels = get_features(dst_train)
# test_features, test_labels = get_features(dst_test)
# model_name= 'CLIP-ViTB32'
# torch.save(train_features, os.path.join(output_dir, f"{model_name}_features.pt"))





# Set paths
images_dir = "path_to_images"  # Replace with your image directory path
output_dir = "/n/fs/dk-diffusion/repos/DeepCore/features/nicospurious/"  # Directory to save features
os.makedirs(output_dir, exist_ok=True)

channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, dst_val= datasets.Nico_95_spurious('~/datasets')
train_loader = torch.utils.data.DataLoader(dst_train, shuffle=False, batch_size=512)

# Transformation for the input images
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define feature extraction function
def save_features(model, model_name, device="cuda"):
    model = model.to(device).eval()
    features = []
    with torch.no_grad():
        for inputs, labels, c_labels, _ in tqdm(train_loader):
            image = inputs.to(device)
            feature = model(image)#.cpu().squeeze(0)
            features.append(feature)
        # features=np.array(features, dtype=object)
        features=torch.cat(features).cpu().numpy()
    torch.save(features, os.path.join(output_dir, f"{model_name}_features.pt"))

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# **Model 1**: RN50 - Imagenet1k
rn50_1k = resnet50(pretrained=True)
rn50_1k = torch.nn.Sequential(*list(rn50_1k.children())[:-1])  # Remove the classifier
save_features(rn50_1k,  "RN50_Imagenet1k", device)

# # **Model 2**: ViT - Imagenet1k
# vit_1k = vit_b_16(pretrained=True)
# vit_1k.heads = torch.nn.Identity()  # Remove the classification head
# save_features(vit_1k, "ViT_Imagenet1k", device)

# # **Model 3**: ViT - Imagenet1k
# vit_1k = vit_b_32(pretrained=True)
# vit_1k.heads = torch.nn.Identity()  # Remove the classification head
# save_features(vit_1k, "ViTb32_Imagenet1k", device)

# # vit_1k = vit_l_16(pretrained=True)
# # vit_1k.heads = torch.nn.Identity()  # Remove the classification head
# # save_features(vit_1k, "ViTl16_Imagenet1k", device)

# # vit_1k = vit_l_32(pretrained=True)
# # vit_1k.heads = torch.nn.Identity()  # Remove the classification head
# # save_features(vit_1k, "ViTl32_Imagenet1k", device)

# dinov2_vitb14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
# dinov2_vitb14.heads = torch.nn.Identity()  # Remove the classification head
# save_features(dinov2_vitb14, "dinov2_vitb14", device)


# import os
# import clip
# import torch
# import numpy as np
# from sklearn.linear_model import LogisticRegression
# from torch.utils.data import DataLoader
# from torchvision.datasets import CIFAR100
# from tqdm import tqdm

# # Load the model
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load('ViT-B/32', device)

# def get_features(dataset):
#     all_features = []
#     all_labels = []
    
#     with torch.no_grad():
#         for images, labels, _, _ in tqdm(DataLoader(dataset, batch_size=100)):
#             features = model.encode_image(images.to(device))

#             all_features.append(features)
#             all_labels.append(labels)

#     return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()

# # Calculate the image features
# train_features, train_labels = get_features(dst_train)
# test_features, test_labels = get_features(dst_test)
# model_name= 'CLIP-ViTB32'
# torch.save(train_features, os.path.join(output_dir, f"{model_name}_features.pt"))

