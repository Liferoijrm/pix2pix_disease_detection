import sys
import os

# Nota: Você deve ter os arquivos 'models', 'options', etc., importáveis.
# Adiciona o diretório pai (raiz do projeto) ao caminho de busca do Python
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import cv2
import torch
from skimage.color import rgb2lab
from models import create_model
from options.test_options import TestOptions

# --- CONFIGURAÇÕES ---
MODEL_NAME = 'final_model_15'
CHECKPOINTS_DIR = '../checkpoints'
IMAGE_SIZE = 256

# ** MODIFIQUE ESTES CAMINHOS **
# 1. Caminho para a imagem de entrada que você quer reconstruir
INPUT_IMAGE_PATH = '../datasets/leaf_disease_detection/test/saudaveis/leaf a13-a15 ab_1.jpg' 
# 2. Caminho e nome do arquivo de saída (será usado para nomear a subpasta)
OUTPUT_FILENAME_BASE = os.path.basename(INPUT_IMAGE_PATH).split('.')[0] # Ex: a976-979 ab_2

# --- FUNÇÕES DE UTILIDADE (setup_options e numpy_rgb_to_lab_tensor permanecem iguais) ---

def setup_options():
    # 1. Cria a instância de opções
    opt = TestOptions().parse()

    # 2. Força a definição dos parâmetros críticos do seu modelo treinado
    opt.dataroot = '../datasets/leaf_disease_detection' # Requerido pelo parser
    opt.name = MODEL_NAME
    opt.checkpoints_dir = CHECKPOINTS_DIR

    opt.epoch = '15'
    
    # Parâmetros de Arquitetura
    opt.model = 'colorization'
    opt.dataset_mode = 'colorization'
    opt.netG = 'unet_256'
    opt.norm = 'batch'
    opt.input_nc = 1
    opt.output_nc = 2
    
    # Parâmetros de Execução
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
    
    return opt

def numpy_rgb_to_lab_tensor(img_rgb_np, device):
    """
    Converte uma imagem numpy RGB (0-255) para um tensor L (normalizado) 
    e um tensor ab (Ground Truth, normalizado).
    """
    lab_np = rgb2lab(img_rgb_np / 255.0)
    lab_tensor = torch.from_numpy(lab_np.astype(np.float32)).to(device).permute(2, 0, 1)

    L_norm = lab_tensor[0:1, :, :] / 50.0 - 1.0
    ab_norm = lab_tensor[1:3, :, :] / 110.0
    
    return L_norm, ab_norm

def reconstruct_image(img_path, output_base_name, model):
    """
    Carrega, reconstrói e salva três versões da imagem em um diretório dedicado.
    """
    print(f"Processando: {img_path}")
    
    # --- 1. CONFIGURAÇÃO DE SAÍDA ---
    output_dir_root = 'pix2pix_reconstructions'
    output_folder = os.path.join(output_dir_root, f'pix2pix_reconstruction_{output_base_name}')
    os.makedirs(output_folder, exist_ok=True)
    print(f"Resultados serão salvos em: {output_folder}")
    
    # --- 2. CARREGAMENTO E PRÉ-PROCESSAMENTO ---
    img_color_orig = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img_color_orig is None:
        print(f"Erro: Não foi possível carregar a imagem em {img_path}")
        return

    # Redimensiona e converte para RGB (OpenCV usa BGR por padrão)
    img_color_orig_rgb = cv2.cvtColor(img_color_orig, cv2.COLOR_BGR2RGB)
    img_color_orig_256 = cv2.resize(img_color_orig_rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)

    # Prepara Tensors
    L_tensor, ab_gt_tensor = numpy_rgb_to_lab_tensor(img_color_orig_256, model.device)
    
    # L_tensor é o 'A' (entrada cinza)
    input_tensor = L_tensor.unsqueeze(0) 
    real_B = ab_gt_tensor.unsqueeze(0) 

    # --- 3. EXECUÇÃO DO MODELO ---
    with torch.no_grad():
        model.set_input({'A': input_tensor, 'B': real_B, 'A_paths': [img_path]}) 
        model.forward() # Executa G(A) -> fake_B

        # 4. Converte Lab (L real, ab falso) para RGB (NumPy)
        reconstructed_rgb_np = model.lab2rgb(model.real_A, model.fake_B) 
        
    # --- 5. SALVAMENTO DAS IMAGENS ---
    
    # A. Imagem Reconstruída (RGB -> BGR para OpenCV)
    reconstructed_bgr = cv2.cvtColor(reconstructed_rgb_np.astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(output_folder, 'reconstructed_rgb_256.png'), reconstructed_bgr)

    # B. Imagem Original Redimensionada (RGB -> BGR para OpenCV)
    orig_bgr_256 = cv2.cvtColor(img_color_orig_256.astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(output_folder, 'original_rgb_256.png'), orig_bgr_256)

    # C. Imagem em Escala de Cinza (L Channel)
    
    # 1. Denormaliza o L-Channel (real_A é o L-channel de entrada)
    # Remove a dimensão do batch (0) e a dimensão do canal (0) com .squeeze().
    L_tensor_denorm = model.real_A.squeeze().cpu().numpy() 
    
    # 2. Converte L (normalizado) de volta para a escala Lab original (0, 100)
    # L_denorm agora é um array 2D (256, 256)
    L_denorm = (L_tensor_denorm + 1.0) * 50.0 

    # 3. Mapeia 0-100 para a escala de 8-bit (0-255)
    L_grayscale = (L_denorm * 2.55).astype(np.uint8) 
    
    # 4. Converte para BGR para salvar
    # Agora cv2.cvtColor receberá um array 2D válido para GRAY
    L_grayscale_bgr = cv2.cvtColor(L_grayscale, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(os.path.join(output_folder, 'input_grayscale_256.png'), L_grayscale_bgr)

# --- EXECUÇÃO PRINCIPAL ---

if __name__ == '__main__':
    opt = setup_options()
    model = create_model(opt)
    model.setup(opt)
    model.eval()
    
    reconstruct_image(INPUT_IMAGE_PATH, OUTPUT_FILENAME_BASE, model)