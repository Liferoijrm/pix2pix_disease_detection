#!/usr/bin/env python3
"""
Grad-CAM em todas as camadas conv do Discriminador (PatchGAN) — comparação lado a lado.

Ajuste WEIGHT_PATH_D, IMAGE_A_PATH, IMAGE_B_PATH e IMG_SIZE antes de rodar.
Salva: gradcam_all_layers_D_100.png
"""
import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# ---------------- CONFIG ----------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Ajuste aqui: checkpoint do D (época 100)
WEIGHT_PATH_D = '../checkpoints/final_model_15/15_net_D.pth'

# Imagem A (entrada L) e B (GT color) correspondentes.
IMAGE_A_PATH = '../datasets/leaf_disease_detection/test/saudaveis/leaf a13-a15 ab_1.jpg '  # L ou A
IMAGE_B_PATH = None  # se tiver GT color correspondente coloque o caminho; se None faremos fallback

IMG_SIZE = 256
OUTPUT_FILE = 'gradcam_all_layers_D_15.png'

# ---------------- util ----------------
def load_rgb(path, size):
    img = Image.open(path).convert('RGB').resize((size, size))
    return img, np.array(img)

def load_gray(path, size):
    img = Image.open(path).convert('L').resize((size, size))
    return img, np.array(img)

def to_tensor_rgb(img_pil, size):
    tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),               # 0..1
        transforms.Normalize([0.5]*3, [0.5]*3)  # -> -1..1
    ])
    return tf(img_pil).unsqueeze(0)  # [1,3,H,W]

def to_tensor_gray(img_pil, size):
    tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),               # 0..1
        transforms.Normalize([0.5], [0.5])   # -> -1..1
    ])
    return tf(img_pil).unsqueeze(0)  # [1,1,H,W]

# ---------------- GradCAM (robusto) ----------------
class GradCAM:
    def __init__(self, model, target_layer_name='default', device=None):
        self.model = model
        self.device = device or torch.device('cpu')
        self.activations = None
        self.hook_handles = []
        self.target_layer_name = target_layer_name
        self.target_layer = self._locate_layer(target_layer_name)
        if self.target_layer is None:
            raise RuntimeError(f"Target layer '{target_layer_name}' not found.")
        self.hook_handles.append(self.target_layer.register_forward_hook(self._forward_hook))

    def _locate_layer(self, name):
        # default: last conv/convtranspose found
        if name == 'default':
            last = None
            for n,m in self.model.named_modules():
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                    last = m
            return last
        # try dotted path or numeric index
        try:
            idx = int(name)
            if hasattr(self.model, 'model') and isinstance(self.model.model, nn.Sequential):
                return self.model.model[idx]
        except Exception:
            pass
        parts = str(name).split('.')
        cur = self.model
        try:
            for p in parts:
                if p.isdigit():
                    cur = cur[int(p)]
                else:
                    cur = getattr(cur, p)
            return cur
        except Exception:
            return None

    def _forward_hook(self, module, inp, out):
        # keep activations (with grad graph) for autograd.grad
        self.activations = out

    def remove_hooks(self):
        for h in self.hook_handles:
            try: h.remove()
            except Exception: pass
        self.hook_handles = []

    def __call__(self, input_tensor, loss_scalar):
        if self.activations is None:
            raise RuntimeError("Forward hook didn't record activation. Make sure forward() ran AFTER constructing GradCAM.")
        self.model.zero_grad()
        grads = torch.autograd.grad(outputs=loss_scalar, inputs=self.activations,
                                    retain_graph=True, create_graph=False)[0]
        weights = grads.mean(dim=(2,3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        cam_min = cam.view(cam.shape[0], -1).min(dim=1)[0].view(-1,1,1,1)
        cam_max = cam.view(cam.shape[0], -1).max(dim=1)[0].view(-1,1,1,1)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam.detach()

# ---------------- load models ----------------
networks = None
try:
    from models import networks as networks_module
    networks = networks_module
except Exception:
    networks = None
    print("Aviso: não foi possível importar 'models.networks' ou 'networks'. Garanta que esteja no PATH.")

def safe_load_state(model, path, device):
    sd = torch.load(path, map_location=device)
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    try:
        model.load_state_dict(sd)
    except Exception:
        new_sd = {}
        for k,v in sd.items():
            nk = k.replace('module.', '') if isinstance(k,str) else k
            new_sd[nk] = v
        model.load_state_dict(new_sd, strict=False)
    return model

# ---------------- list conv layers helper ----------------
def list_conv_layers(model):
    convs = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            if name == '':
                continue
            convs.append((name, module))
    return convs

# ---------------- main ----------------
def main():
    print("Device:", DEVICE)
    if not os.path.exists(IMAGE_A_PATH):
        raise FileNotFoundError("IMAGE_A_PATH not found: " + IMAGE_A_PATH)

    # load inputs
    img_A_pil, A_np = load_gray(IMAGE_A_PATH, IMG_SIZE)
    tensor_A = to_tensor_gray(img_A_pil, IMG_SIZE).to(DEVICE)  # [1,1,H,W]

    has_B = False
    if IMAGE_B_PATH and os.path.exists(IMAGE_B_PATH):
        img_B_pil, B_np = load_rgb(IMAGE_B_PATH, IMG_SIZE)
        tensor_B = to_tensor_rgb(img_B_pil, IMG_SIZE).to(DEVICE)  # [1,3,H,W]
        has_B = True
        print("GT B provided.")
    else:
        # fallback: duplicate grayscale as 3-channel (not ideal but usable)
        print("GT B not provided — using grayscale->RGB fallback for B.")
        B_np = np.stack([A_np]*3, axis=2)
        img_B_pil = Image.fromarray(B_np)
        tensor_B = to_tensor_rgb(img_B_pil, IMG_SIZE).to(DEVICE)

    # instantiate discriminator (try common signatures)
    if networks is None:
        raise RuntimeError("Não foi possível importar networks. Garanta o repositório no PATH.")
    D = None
    # try common call patterns
    tried = []
    try:
        D = networks.define_D(input_nc=3, ndf=64, netD='basic', n_layers_D=3, norm='batch')
        tried.append("define_D(input_nc=3, netD='basic')")
    except Exception:
        try:
            D = networks.define_D()
            tried.append("define_D()")
        except Exception as e:
            print("Falha ao instanciar D:", e)
            raise RuntimeError("Não foi possível instanciar Discriminador automaticamente.")

    # load weights
    if not os.path.exists(WEIGHT_PATH_D):
        raise FileNotFoundError("WEIGHT_PATH_D not found: " + WEIGHT_PATH_D)
    D = safe_load_state(D, WEIGHT_PATH_D, DEVICE)
    D.to(DEVICE).eval()
    print("Discriminator loaded. instantiation attempts:", tried)

    # determine expected input channels by inspecting first conv if possible
    first_conv = None
    for name,module in D.named_modules():
        if isinstance(module, nn.Conv2d):
            first_conv = module
            first_conv_name = name
            break
    if first_conv is None:
        raise RuntimeError("Não encontrei Conv2d no discriminador.")
    in_ch = first_conv.weight.shape[1]
    print("Discriminator's first conv expects", in_ch, "input channels (weight shape:", first_conv.weight.shape, ")")

    # prepare input for discriminator: concat A and B suitably to match channels
    # common pix2pix: input = concat(A,B) with channels = input_nc + output_nc
    # If in_ch == 3 and A is grayscale => use B only
    if in_ch == 1:
        img_for_d = tensor_A
    elif in_ch == 2:
        # rare, but might be L + a or something; pack as [A, B_channel0] if possible
        # fallback: take first channel of tensor_B
        img_for_d = torch.cat([tensor_A, tensor_B[:,0:1,:,:]], dim=1)
    elif in_ch == 3:
        # assume discriminator expects RGB image only (B)
        img_for_d = tensor_B
    else:
        # assume concat; try A(1) + B( in_ch-1 )
        need = in_ch - 1
        # if B has >= need channels use first need channels; else tile
        b_ch = tensor_B.shape[1]
        if b_ch >= need:
            img_for_d = torch.cat([tensor_A, tensor_B[:, :need, :, :]], dim=1)
        else:
            # tile B channels to reach need
            pads = []
            while sum([p.shape[1] for p in pads]) + b_ch < need:
                pads.append(tensor_B)
            cat_b = torch.cat([tensor_B] + pads, dim=1)[:, :need, :, :]
            img_for_d = torch.cat([tensor_A, cat_b], dim=1)

    img_for_d = img_for_d.to(DEVICE)
    img_for_d.requires_grad_(True)

    # collect conv layers
    convs = list_conv_layers(D)
    print(f"Found {len(convs)} conv/convtranspose layers in D. Will attempt Grad-CAM on each.")

    overlays = []
    titles = []

    for (layer_name, layer_module) in convs:
        try:
            print("Processing D layer:", layer_name)
            gcam = GradCAM(D, layer_name, device=DEVICE)
            # forward after gcam instantiation
            out = D(img_for_d)
            # normalize shape -> [B,1,Hf,Wf] if needed
            if out.dim() == 3:
                out = out.unsqueeze(1)
            # scalar score
            score = out.mean()
            cam_map = gcam(img_for_d, score)  # 1x1xHxW
            gcam.remove_hooks()

            cam_np = cam_map.squeeze(0).squeeze(0).cpu().numpy()  # HxW

            # prepare display (use B RGB for overlay)
            display_rgb = None
            if has_B:
                display_rgb = np.array(img_B_pil.resize((IMG_SIZE, IMG_SIZE)))
            else:
                display_rgb = np.array(img_B_pil)

            cmap = plt.get_cmap('jet')
            heat = cmap(cam_np)[:,:,:3]
            heat_u8 = np.uint8(heat * 255)
            overlay = (display_rgb * 0.6 + heat_u8 * 0.4).astype(np.uint8)

            overlays.append(overlay)
            titles.append(layer_name)
        except Exception as e:
            print(f"Layer {layer_name} skipped due to error: {e}")
            continue

    if len(overlays) == 0:
        print("No overlays generated.")
        return

    # grid
    max_cols = 6
    n = len(overlays)
    cols = min(max_cols, n)
    rows = math.ceil(n / cols)

    fig_w = cols * 3
    fig_h = rows * 3
    fig, axs = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    axs = np.array(axs).reshape(rows, cols)

    idx = 0
    for r in range(rows):
        for c in range(cols):
            ax = axs[r,c]
            if idx < n:
                ax.imshow(overlays[idx])
                ax.set_title(titles[idx], fontsize=8)
            else:
                ax.axis('off')
            ax.axis('off')
            idx += 1

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=200)
    print("Saved visualization to", OUTPUT_FILE)
    plt.show()

if __name__ == '__main__':
    main()
