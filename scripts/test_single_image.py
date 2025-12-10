import sys
import os

# Adiciona o diretório pai (raiz do projeto) ao caminho de busca do Python
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '..')
sys.path.append(project_root)

import argparse # Adicionado para ler argumentos
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from skimage import color 
from glob import glob
from options.test_options import TestOptions
from models import create_model

# --- CONFIGURAÇÕES ---
MODEL_NAME = 'final_model_15'
CHECKPOINTS_DIR = '../checkpoints'
IMAGE_SIZE = 256
DEVICE = None 

INPUT_IMAGE_PATH_DEFAULT = '../datasets/leaf_disease_detection/test/saudaveis/leaf a13-a15 ab_1.jpg' 
OUTPUT_DIR_ROOT = '../results/test_single_leaf' 

def setup_options():
    global DEVICE
    opt = TestOptions().parse()
    opt.dataroot = '../datasets/leaf_disease_detection'
    opt.name = MODEL_NAME
    opt.checkpoints_dir = CHECKPOINTS_DIR
    opt.epoch = '15'
    opt.model = 'colorization'
    opt.dataset_mode = 'colorization'
    opt.netG = 'unet_256'
    opt.norm = 'batch'
    opt.input_nc = 1
    opt.output_nc = 2
    opt.num_threads = 0
    opt.batch_size = 1
    opt.no_flip = True
    opt.eval = True 
    opt.phase = 'test'
    opt.load_size = IMAGE_SIZE
    opt.crop_size = IMAGE_SIZE

    if torch.cuda.is_available():
        opt.device = torch.device('cuda:0')
        opt.gpu_ids = [0]
    else:
        opt.device = torch.device('cpu')
        opt.gpu_ids = []
    
    DEVICE = opt.device
    return opt

def numpy_rgb_to_lab_tensor(img_rgb_np, device):
    """ Converte uma imagem numpy RGB (0-255) para tensores L e ab normalizados. """
    lab_np = color.rgb2lab(img_rgb_np / 255.0)
    lab_tensor = torch.from_numpy(lab_np.astype(np.float32)).to(device).permute(2, 0, 1)

    L_norm = lab_tensor[0:1, :, :] / 50.0 - 1.0
    ab_norm = lab_tensor[1:3, :, :] / 110.0
    
    return L_norm, ab_norm

def process_anomaly(original_rgb_np, reconstructed_rgb_np):
    """ 
    Calcula a diferença de cor CIEDE2000 entre a imagem original e a reconstruída.
    Retorna a média e o mapa de erro.
    """
    original_lab = color.rgb2lab(original_rgb_np / 255.0)
    reconstructed_lab = color.rgb2lab(reconstructed_rgb_np / 255.0)
    
    delta_e_map = color.deltaE_ciede2000(original_lab, reconstructed_lab)
    score_mean = np.mean(delta_e_map)
    
    return score_mean, delta_e_map

def plot_single_sample_pdf(output_path, img_gray, img_orig, img_recon, diff_map, 
                          filename, score_ciede, is_disease, threshold=1.7391):
    """
    Cria um PDF com as 4 imagens e o resumo das métricas.
    """
    status_real = "DOENTE" if is_disease else "SAUDÁVEL"
    pred_label = 1 if score_ciede > threshold else 0
    status_pred = "DOENTE" if pred_label == 1 else "SAUDÁVEL"
    
    info_text = (f"Arq: {filename} | Score CIEDE: {score_ciede:.4f} | Limiar: {threshold:.4f}\n"
                 f"Predito: {status_pred} | Rótulo Real: {status_real}")
    
    fig, ax = plt.subplots(1, 4, figsize=(16, 5))
    plt.subplots_adjust(top=0.75, wspace=0.3) 
    
    ax[0].imshow(img_gray, cmap='gray'); ax[0].set_title("1. Entrada (L Channel)"); ax[0].axis('off')
    ax[1].imshow(img_orig); ax[1].set_title("2. Original (RGB)"); ax[1].axis('off')
    ax[2].imshow(img_recon); ax[2].set_title("3. Reconstruída (Fake RGB)"); ax[2].axis('off')

    im = ax[3].imshow(diff_map, cmap='jet', vmin=0, vmax=20)
    ax[3].set_title("4. Mapa de Erro CIEDE2000"); ax[3].axis('off')
    
    fig.colorbar(im, ax=ax[3], fraction=0.046, pad=0.04)
    fig.suptitle(info_text, fontsize=14, y=0.92, fontweight='bold')
    
    with PdfPages(output_path) as pdf:
        pdf.savefig(fig)
    
    plt.close(fig)

def reconstruct_image(img_path, output_base_name, model):
    """
    Carrega, reconstrói, calcula CIEDE, salva imagens e gera o relatório PDF.
    """
    print(f"Processando: {img_path}")
    
    # --- 1. CONFIGURAÇÃO DE SAÍDA ---
    output_folder = os.path.join(OUTPUT_DIR_ROOT, f'reconstruction_{output_base_name}')
    os.makedirs(output_folder, exist_ok=True)
    print(f"Resultados serão salvos em: {os.path.abspath(output_folder)}")
    
    # Determinar Rótulo Real baseado no caminho
    is_disease = 1 if 'doentes' in img_path.lower() else 0

    # --- 2. CARREGAMENTO E PRÉ-PROCESSAMENTO ---
    img_color_orig = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img_color_orig is None:
        print(f"Erro: Não foi possível carregar a imagem em {img_path}")
        return

    img_color_orig_rgb = cv2.cvtColor(img_color_orig, cv2.COLOR_BGR2RGB)
    img_color_orig_256 = cv2.resize(img_color_orig_rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)

    L_tensor, ab_gt_tensor = numpy_rgb_to_lab_tensor(img_color_orig_256, model.device)
    
    input_tensor = L_tensor.unsqueeze(0) 
    real_B = ab_gt_tensor.unsqueeze(0) 

    # --- 3. EXECUÇÃO DO MODELO ---
    with torch.no_grad():
        model.set_input({'A': input_tensor, 'B': real_B, 'A_paths': [img_path]}) 
        model.forward()
        reconstructed_rgb_np = model.lab2rgb(model.real_A, model.fake_B) 
        
    # --- 4. CÁLCULO CIEDE E HEATMAP ---
    score_ciede, diff_map = process_anomaly(img_color_orig_256, reconstructed_rgb_np)
    print(f"Score CIEDE2000 Médio: {score_ciede:.4f}")

    # --- 5. SALVANDO ARQUIVOS INDIVIDUAIS ---
    
    # A. Reconstruída
    reconstructed_bgr = cv2.cvtColor(reconstructed_rgb_np.astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(output_folder, 'reconstructed_rgb_256.png'), reconstructed_bgr)

    # B. Original
    orig_bgr_256 = cv2.cvtColor(img_color_orig_256.astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(output_folder, 'original_rgb_256.png'), orig_bgr_256)

    # C. Grayscale (L Channel)
    L_tensor_denorm = model.real_A.squeeze().cpu().numpy() 
    L_denorm = (L_tensor_denorm + 1.0) * 50.0 
    L_grayscale = (L_denorm * 2.55).astype(np.uint8) 
    L_grayscale_bgr = cv2.cvtColor(L_grayscale, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(os.path.join(output_folder, 'input_grayscale_256.png'), L_grayscale_bgr)
    
    # D. Heatmap
    plt.figure(figsize=(5, 4))
    plt.title(f"Mapa de Erro CIEDE2000 (Score: {score_ciede:.4f})")
    plt.axis('off')
    im = plt.imshow(diff_map, cmap='jet', vmin=0, vmax=20)
    plt.colorbar(im, fraction=0.046, pad=0.04, label='Delta E CIEDE2000')
    plt.savefig(os.path.join(output_folder, 'ciede_heatmap_256.png'), bbox_inches='tight')
    plt.close()


    # --- 6. GERAR RELATÓRIO PDF ---
    pdf_path = os.path.join(output_folder, f'{output_base_name}_report.pdf')
    plot_single_sample_pdf(
        output_path=pdf_path, 
        img_gray=L_grayscale, 
        img_orig=img_color_orig_256, 
        img_recon=reconstructed_rgb_np.astype(np.uint8), 
        diff_map=diff_map,
        filename=os.path.basename(img_path), 
        score_ciede=score_ciede,
        is_disease=is_disease,
        threshold=1.7391
    )
    print(f"Relatório PDF gerado em: {os.path.abspath(pdf_path)}")

# --- EXECUÇÃO PRINCIPAL MODIFICADA ---

if __name__ == '__main__':
    
    # 1. Cria um parser para o argumento específico do script
    parser = argparse.ArgumentParser(description="Processa uma única imagem com o modelo Pix2Pix.")
    parser.add_argument('--path', type=str, default=INPUT_IMAGE_PATH_DEFAULT, 
                        help='Caminho completo para a imagem de entrada (sobrescreve o default hardcoded).')
    
    # 2. Faz o parsing dos argumentos de linha de comando
    args, unknown = parser.parse_known_args()
    
    # 3. Define as variáveis globais do script com base no argumento (se fornecido)
    input_image_path = args.path
    output_filename_base = os.path.basename(input_image_path).split('.')[0] 
    
    # 4. Configura e Carrega o Modelo
    opt = setup_options()
    model = create_model(opt)
    model.setup(opt)
    model.eval()
    
    # 5. Executa a Reconstrução
    reconstruct_image(input_image_path, output_filename_base, model)