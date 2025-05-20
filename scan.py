import os
import torch.nn as nn
import torch
from PIL import Image
from torchvision import transforms

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_img(img_path, size=512):
    image=Image.open(img_path).convert("RGB")
    transform=transforms.Compose([transforms.Resize(size),transforms.CenterCrop(size),transforms.ToTensor()])
    image=transform(image).unsqueeze(0).to(device)
    return image

def save_img(tensor, path):
    tensor=tensor.clone().detach().squeeze(0)
    tensor=transforms.ToPILImage()(Image.clamp(0,1))
    tensor.save(path)

def adain(con_feat, style_feat):
    size=con_feat.size()
    style_mean, style_std=calc_mean_std(style_feat)
    con_mean, con_std=calc_mean_std(con_feat)
    normalized_feat=(con_feat-con_mean.expand(size))/con_std.expand(size)
    return normalized_feat*style_std.expand(size)+style_mean.expand(size)

def calc_mean_std(feat, eps=1e-5):
    size=feat.size()
    N,C =size[:2]
    feat_var=feat.view(N,C,-1).var(dim=2)+eps
    feat_std=feat_var.sqrt().view(N,C,1,1)
    feat_mean=feat.view(N,C,-1).mean(dim=2).view(N,C,1,1)
    return feat_mean, feat_std

class VGGE(nn.module):
    def __init__(self, vgg):
        super(VGGE,self).__init__()
        self.enc=nn.Sequential(*list(vgg.children())[:21])

    def forward(self, x):
        x=self.enc(x)
        return x
   
def get_decoder():
    decoder=torch.load("models/decoder.pth")
    return decoder

def get_vgg():
    vgg=torch.load("models/vgg_normalised.pth")
    return VGGE(vgg)

def stylize(content_path, style_path, output_path="output.jpg", a=1.0):
    con=load_img(content_path)
    style=load_img(style_path)
    decoder=get_decoder().to(device).eval()
    vgg=get_vgg().to(device).eval()
    with torch.no_grad():
        con_feat=vgg(con)
        style_feat=vgg(style)
        t=adain(con_feat, style_feat)
        t=a*t+(1-a)*con_feat
        output=decoder(t)
    save_img(output, output_path)
    print(f"Stylized image saved to {output_path}")