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
from glob import glob
from matplotlib.backends.backend_pdf import PdfPages 

# Adiciona o diretório pai ao PATH, se necessário, para importar 'models'
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# ---------------- CONFIG ----------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# checkpoint do D (época 15)
WEIGHT_PATH_D = '../checkpoints/final_model_15/15_net_D.pth'

# Caminho raiz para o dataset de teste
DATASET_ROOT = '../datasets/leaf_disease_detection/test'

IMG_SIZE = 256

# Diretório raiz para a saída
OUTPUT_ROOT = '../results/Grad-CAM_layers'
OUTPUT_DIR_IMG = {'saudaveis': os.path.join(OUTPUT_ROOT, 'SAUDAVEIS', 'imagens'),
                  'doentes': os.path.join(OUTPUT_ROOT, 'DOENTES', 'imagens')}
OUTPUT_DIR_PDF = {'saudaveis': os.path.join(OUTPUT_ROOT, 'SAUDAVEIS', 'pdf'),
                  'doentes': os.path.join(OUTPUT_ROOT, 'DOENTES', 'pdf')}
FINAL_REPORT_PATH = os.path.join(OUTPUT_ROOT, 'RELATORIO_GRADCAM_COMPLETO.pdf')


# ---------------- util ----------------
def load_rgb(path, size):
    img = Image.open(path.strip()).convert('RGB').resize((size, size))
    return img, np.array(img)

def load_gray(path, size):
    img = Image.open(path.strip()).convert('L').resize((size, size))
    return img, np.array(img)

def to_tensor_rgb(img_pil, size):
    tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),              
        transforms.Normalize([0.5]*3, [0.5]*3)  
    ])
    return tf(img_pil).unsqueeze(0)  

def to_tensor_gray(img_pil, size):
    tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),              
        transforms.Normalize([0.5], [0.5])    
    ])
    return tf(img_pil).unsqueeze(0)  

# ---------------- GradCAM (robusto) ----------------
class GradCAM:
    # (A classe GradCAM não foi alterada)
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
        if name == 'default':
            last = None
            for n,m in self.model.named_modules():
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                    last = m
            return last
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

# ---------------- load models / list layers ----------------
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

def list_conv_layers(model):
    convs = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            if name == '':
                continue
            convs.append((name, module))
    return convs

# ---------------- main processing logic for a single image ----------------
def process_single_image_gradcam(image_A_path, D, convs, in_ch):
    """
    Processa uma única imagem com Grad-CAM.
    Salva PNG e retorna o objeto Figura e o caminho do PNG.
    """
    category = 'saudaveis' if 'saudaveis' in image_A_path.lower() else 'doentes'
    
    try:
        if not os.path.exists(image_A_path):
            print(f"File not found: {image_A_path}. Skipping.")
            return None, None

        img_A_pil, A_np = load_gray(image_A_path, IMG_SIZE)
        tensor_A = to_tensor_gray(img_A_pil, IMG_SIZE).to(DEVICE)

        B_np = np.stack([A_np]*3, axis=2)
        img_B_pil = Image.fromarray(B_np)
        tensor_B = to_tensor_rgb(img_B_pil, IMG_SIZE).to(DEVICE)

        # 2. prepare input for discriminator
        img_for_d = None
        if in_ch == 1:
            img_for_d = tensor_A
        elif in_ch == 2:
            img_for_d = torch.cat([tensor_A, tensor_B[:,0:1,:,:]], dim=1)
        elif in_ch == 3:
            img_for_d = tensor_B
        else:
            need = in_ch - 1
            b_ch = tensor_B.shape[1]
            if b_ch >= need:
                img_for_d = torch.cat([tensor_A, tensor_B[:, :need, :, :]], dim=1)
            else:
                pads = []
                while sum([p.shape[1] for p in pads]) + b_ch < need:
                    pads.append(tensor_B)
                cat_b = torch.cat([tensor_B] + pads, dim=1)[:, :need, :, :]
                img_for_d = torch.cat([tensor_A, cat_b], dim=1)

        img_for_d = img_for_d.to(DEVICE)
        img_for_d.requires_grad_(True)

        overlays = []
        titles = []

        # 3. Grad-CAM for each conv layer
        for (layer_name, layer_module) in convs:
            try:
                gcam = GradCAM(D, layer_name, device=DEVICE)
                out = D(img_for_d) 
                
                if out.dim() == 3:
                    out = out.unsqueeze(1)
                
                score = out.mean() 
                cam_map = gcam(img_for_d, score)
                gcam.remove_hooks()

                cam_np = cam_map.squeeze(0).squeeze(0).cpu().numpy()
                display_rgb = np.array(img_B_pil.resize((IMG_SIZE, IMG_SIZE)))
                cmap = plt.get_cmap('jet')
                heat = cmap(cam_np)[:,:,:3]
                heat_u8 = np.uint8(heat * 255)
                overlay = (display_rgb * 0.6 + heat_u8 * 0.4).astype(np.uint8)

                overlays.append(overlay)
                titles.append(layer_name)
            except Exception as e:
                print(f"  Layer {layer_name} skipped for {image_A_path} due to error: {e}")
                continue

        if len(overlays) == 0:
            print(f"No overlays generated for {image_A_path}.")
            return None, None

        # 4. Grid visualization (Ajuste de espaçamento)
        max_cols = 6
        n = len(overlays)
        cols = min(max_cols, n)
        rows = math.ceil(n / cols)

        fig_w = cols * 3.5 # Aumenta um pouco a largura
        fig_h = rows * 3.5 # Aumenta um pouco a altura
        fig, axs = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
        axs = np.array(axs).reshape(rows, cols)

        idx = 0
        for r in range(rows):
            for c in range(cols):
                ax = axs[r,c]
                if idx < n:
                    ax.imshow(overlays[idx])
                    # Título da camada com espaçamento ajustado
                    ax.set_title(titles[idx], fontsize=8, pad=2) 
                ax.axis('off')
                idx += 1

        # Ajuste de layout para evitar sobreposição do título principal e subplots
        plt.tight_layout(rect=[0, 0, 1, 0.96]) 
        plt.suptitle(f"{category.upper()}: {os.path.basename(image_A_path)}", fontsize=12, y=0.99)
        
        # 5. Saving PNG (O PDF será salvo na função de relatório final)
        safe_filename = os.path.basename(image_A_path).replace('.', '_')
        output_file_png = os.path.join(OUTPUT_DIR_IMG[category], f'gradcam_{safe_filename}.png')
        
        fig.savefig(output_file_png, dpi=200)

        print(f"  Saved PNG to {output_file_png}")
        
        # Retorna a figura para ser incluída no PDF principal
        return fig, output_file_png

    except Exception as e:
        print(f"Critical error processing {image_A_path}: {e}")
        return None, None

def create_final_report_pdf(figures, output_path, D, convs, png_paths_by_category):
    """
    Cria um único arquivo PDF combinando todas as figuras Grad-CAM.
    Salva também os PDFs individuais para a estrutura de pastas.
    """
    if not figures:
        print("No figures to include. Skipping final report.")
        return

    print(f"\n--- Creating Final Report PDF at {output_path} ---")

    # 1. Cria o documento PDF multipágina
    with PdfPages(output_path) as pdf:
        
        # Adiciona uma página de capa simples
        fig_cover, ax_cover = plt.subplots(figsize=(8.5, 11))
        ax_cover.text(0.5, 0.7, "Relatório Grad-CAM (PatchGAN Discriminator)", 
                      fontsize=16, ha='center', va='center')
        ax_cover.text(0.5, 0.6, f"Modelo: {os.path.basename(WEIGHT_PATH_D)}", 
                      fontsize=12, ha='center', va='center')
        ax_cover.text(0.5, 0.55, f"Camadas processadas: {[name for name, _ in convs]}", 
                      fontsize=10, ha='center', va='center', wrap=True)
        ax_cover.text(0.5, 0.45, f"Total de Amostras: {len(figures)}", 
                      fontsize=10, ha='center', va='center')
        ax_cover.axis('off')
        pdf.savefig(fig_cover)
        plt.close(fig_cover)
        
        # 2. Adiciona todas as figuras Grad-CAM coletadas
        for fig in figures:
            pdf.savefig(fig)
            plt.close(fig) # Fecha após salvar para liberar memória
        
        print(f"Relatório final multipágina salvo com sucesso em: {output_path}")
        
    # 3. Salva os PDFs individuais (agora que o relatório principal foi criado)
    for category, paths in png_paths_by_category.items():
        for png_path in paths:
            # O nome do arquivo PNG é reutilizado para o PDF
            base_name = os.path.basename(png_path).replace('.png', '.pdf')
            pdf_path = os.path.join(OUTPUT_DIR_PDF[category], base_name)
            
            # Reabre o PNG para salvá-lo como PDF (mantendo a figura fechada)
            try:
                img_fig = plt.figure(figsize=(8.5, 11))
                img = Image.open(png_path)
                plt.imshow(img)
                plt.axis('off')
                img_fig.savefig(pdf_path, dpi=200)
                plt.close(img_fig)
            except Exception as e:
                print(f"Aviso: Falha ao salvar PDF individual para {png_path}: {e}")

def process_dataset():
    print("Device:", DEVICE)
    
    # 1. Garantir que os diretórios de saída existam
    for d in OUTPUT_DIR_IMG.values():
        os.makedirs(d, exist_ok=True)
    for d in OUTPUT_DIR_PDF.values():
        os.makedirs(d, exist_ok=True)
        
    print(f"Output root directory: {OUTPUT_ROOT}")
    print("Directories created for SAUDAVEIS/DOENTES (imagens/pdf).")

    # 2. Instanciar e carregar o Discriminador (D)
    if networks is None:
        raise RuntimeError("Não foi possível importar networks. Garanta o repositório no PATH.")
    
    D = None
    try:
        # Tenta a assinatura mais comum para um PatchGAN
        D = networks.define_D(input_nc=3, ndf=64, netD='basic', n_layers_D=3, norm='batch')
    except Exception as e:
        print("Falha ao instanciar D:", e)
        raise RuntimeError("Não foi possível instanciar Discriminador automaticamente.")

    if not os.path.exists(WEIGHT_PATH_D):
        raise FileNotFoundError("WEIGHT_PATH_D not found: " + WEIGHT_PATH_D)
    
    D = safe_load_state(D, WEIGHT_PATH_D, DEVICE)
    D.to(DEVICE).eval()
    print("Discriminator loaded.")

    # 3. Encontrar camadas convolucionais
    convs = list_conv_layers(D)
    if not convs:
         raise RuntimeError("Não encontrei camadas conv/convtranspose no Discriminador.")
    print(f"Found {len(convs)} conv/convtranspose layers in D.")

    # 4. Determinar canais de entrada esperados pelo D
    first_conv = None
    for module in D.modules():
        if isinstance(module, nn.Conv2d):
            first_conv = module
            break
    if first_conv is None:
        raise RuntimeError("Não encontrei Conv2d no discriminador.")
    in_ch = first_conv.weight.shape[1]
    print("Discriminator's first conv expects", in_ch, "input channels.")
    
    # 5. Iterar sobre o dataset e coletar figuras e caminhos PNG
    search_path = os.path.join(DATASET_ROOT, '**', '*.jpg')
    image_paths = glob(search_path, recursive=True)
    
    if not image_paths:
        print(f"No images found in {DATASET_ROOT}. Check path and file extensions.")
        return

    print(f"Found {len(image_paths)} images to process.")
    
    all_figures = []
    png_paths_by_category = {'saudaveis': [], 'doentes': []}
    
    for i, path in enumerate(image_paths):
        print(f"\n--- Processing {i+1}/{len(image_paths)}: {path} ---")
        fig, png_path = process_single_image_gradcam(path, D, convs, in_ch)
        
        if fig is not None:
            all_figures.append(fig)
            category = 'saudaveis' if 'saudaveis' in path.lower() else 'doentes'
            if png_path:
                 png_paths_by_category[category].append(png_path)
            
    # 6. Criar Relatório Final PDF (Multipágina) e salvar PDFs individuais
    create_final_report_pdf(all_figures, FINAL_REPORT_PATH, D, convs, png_paths_by_category)


if __name__ == '__main__':
    process_dataset()