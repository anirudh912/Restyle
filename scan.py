import os
import torch.nn as nn
import torch
from PIL import Image
from torchvision import transforms, models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_img(img_path, size=512):
    image = Image.open(img_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor()
    ])
    image = transform(image).unsqueeze(0).to(device)
    return image

def save_img(tensor, path):
    tensor = tensor.clone().detach().squeeze(0)
    tensor = transforms.ToPILImage()(tensor.clamp(0, 1))
    tensor.save(path)

def adain(con_feat, style_feat):
    size = con_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    con_mean, con_std = calc_mean_std(con_feat)
    normalized_feat = (con_feat - con_mean.expand(size)) / con_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

def calc_mean_std(feat, eps=1e-5):
    size = feat.size()
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

class VGGE(nn.Module):
    def __init__(self):
        super(VGGE, self).__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=1),          # Layer 0 (1x1 conv)
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, 64, kernel_size=3),         # Layer 2
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # Layer 4
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # Layer 7
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),# Layer 9
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),# Layer 12
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),# Layer 14
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),# Layer 16
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),# Layer 19
            nn.ReLU(inplace=True)
        )
        state_dict = torch.load("models/vgg_normalised.pth", map_location=device)
        self.enc.load_state_dict(state_dict)

    def forward(self, x):
        return self.enc(x)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.layers = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 256, 3),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, 3),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 128, 3),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 64, 3),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 3, 3),
        )

    def forward(self, x):
        return self.layers(x)

def get_decoder():
    decoder = Decoder()
    st_dict = torch.load("models/decoder.pth", map_location=device)
    fixed_dict = {f"layers.{k}": v for k, v in st_dict.items()}
    decoder.load_state_dict(fixed_dict)
    return decoder

def get_vgg():
    return VGGE()

def stylize(content_path, style_path, output_path="output.jpg", a=1.0):
    con = load_img(content_path)
    style = load_img(style_path, size=con.shape[2])
    decoder = get_decoder().to(device).eval()
    vgg = get_vgg().to(device).eval()
    with torch.no_grad():
        con_feat = vgg(con)
        style_feat = vgg(style)
        t = adain(con_feat, style_feat)
        t = a * t + (1 - a) * con_feat
        output = decoder(t)
    save_img(output, output_path)
    print(f"Stylized image saved to {output_path}")

stylize("base.jpg", "style.jpg", "output.jpg", 1.0)
