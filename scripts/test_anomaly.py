import sys
import os
import numpy as np
import torch
import torch.nn as nn
from skimage import color # Importa skimage.color para rgb2lab e deltaE_ciede2000
# Importa as métricas necessárias para AUC-ROC e para cálculo do limiar e métricas
from sklearn.metrics import (
    roc_auc_score, 
    roc_curve, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score
)
from glob import glob # Para buscar arquivos no dataset

# Adiciona o diretório raiz do PyTorch-CycleGAN-and-pix2pix (o diretório PAI do 'scripts')
# para que os módulos 'models' e 'options' possam ser importados.
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '..')
sys.path.append(project_root)

# Assumindo que essas importações são do seu projeto (necessárias para o modelo)
try:
    from models import create_model
    from options.test_options import TestOptions
except ImportError as e:
    print(f"ERRO CRÍTICO DE IMPORTAÇÃO: {e}")
    print("Verifique se o path do projeto está correto e os módulos 'models' e 'options' existem.")
    sys.exit(1) # Força a saída se não conseguir importar

# --- CONFIGURAÇÕES E VARIÁVEIS GLOBAIS ---
MODEL_NAME = 'final_model_15'
CHECKPOINTS_DIR = '../checkpoints'
DATASET_ROOT = '../datasets/leaf_disease_detection/test'
IMAGE_SIZE = 256
DEVICE = None 


# --- FUNÇÕES DE UTILIDADE ---

def setup_options():
    """Configurações e define o dispositivo."""
    global DEVICE
    
    # Simula a leitura das opções de teste
    opt = TestOptions().parse()
    
    # --- SOBRESCRITA CRÍTICA DE CONFIGURAÇÃO (para Colorization/Anomalia) ---
    opt.name = MODEL_NAME
    opt.model = 'colorization'
    opt.dataset_mode = 'colorization'
    opt.netG = 'unet_256' # CRÍTICO: Garante a arquitetura UNet
    opt.input_nc = 1 # CRÍTICO: Garante 1 canal de entrada (L)
    opt.output_nc = 2 # CRÍTICO: Garante 2 canais de saída (ab)
    
    # **CORREÇÃO CRÍTICA:** Força o tipo de normalização.
    opt.norm = 'batch' 
    
    # ----------------------------------------------------------------------
    # >>> MUDANÇA CRÍTICA AQUI: FORÇA O CARREGAMENTO DA ÉPOCA 100
    # ----------------------------------------------------------------------
    opt.epoch = '15'  # <--- Use o nome do arquivo do checkpoint (ex: '100' para netG_100.pth)
    
    # Outras configurações
    opt.num_threads = 0
    opt.batch_size = 1
    opt.no_flip = True
    opt.checkpoints_dir = CHECKPOINTS_DIR
    opt.eval = True 
    opt.phase = 'test' # Garante a fase correta

    # --- CONFIGURAÇÃO E ATRIBUIÇÃO DO DEVICE ---
    if torch.cuda.is_available():
        # Define o atributo 'device' no objeto opt
        opt.device = torch.device('cuda:0')
        opt.gpu_ids = [0] 
    else:
        # Define o atributo 'device' no objeto opt
        opt.device = torch.device('cpu')
        opt.gpu_ids = []
    
    # Atribui a variável global DEVICE (agora opt.device existe)
    DEVICE = opt.device 
    
    return opt

def numpy_rgb_to_lab_tensor(img_rgb_np):
    """
    Converte uma imagem numpy RGB (0-255) para um tensor Lab normalizado 
    (L para input A e ab para target B).
    """
    # 1. Converte para Lab (skimage)
    lab_np = color.rgb2lab(img_rgb_np / 255.0)
    
    # 2. Converte para PyTorch Tensor [H, W, C] -> [C, H, W]
    lab_tensor = torch.from_numpy(lab_np.astype(np.float32)).to(DEVICE).permute(2, 0, 1)

    # 3. Aplica a normalização padrão do Pix2pix Colorization:
    # L (A): [0, 100] -> [-1, 1]
    # a, b (B): [-128, 127] -> [-1, 1]
    L_norm = lab_tensor[0:1, :, :] / 50.0 - 1.0
    ab_norm = lab_tensor[1:3, :, :] / 110.0
    
    return L_norm.unsqueeze(0), ab_norm.unsqueeze(0) # Retorna BxCxHxW


def calculate_anomaly_score_mean(original_rgb_np, reconstructed_rgb_np):
    """
    Calcula o score de anomalia (média do CIEDE2000) entre duas imagens RGB NumPy (0-255).
    """
    # Normaliza de 0-255 para 0-1
    original_lab = color.rgb2lab(original_rgb_np / 255.0)
    reconstructed_lab = color.rgb2lab(reconstructed_rgb_np / 255.0)
    
    # Calcula a diferença pixel a pixel
    delta_e = color.deltaE_ciede2000(original_lab, reconstructed_lab)
    
    # O score é a média global da diferença
    return np.mean(delta_e)

# --- FUNÇÃO PRINCIPAL DE PROCESSAMENTO ---

def run_metric_evaluation():
    opt = setup_options()
    
    # 1. Cria e Carrega o Modelo
    print(f"Criando modelo: {opt.model} (NetG: {opt.netG}, Input: {opt.input_nc}, Output: {opt.output_nc}, Norm: {opt.norm})")
    model = create_model(opt)
    model.setup(opt) 
    model.eval()

    all_scores = []
    all_labels = [] # 0 = Saudável, 1 = Doente

    # 2. Listagem dos Arquivos
    healthy_dir = os.path.join(DATASET_ROOT, 'saudaveis')
    disease_dir = os.path.join(DATASET_ROOT, 'doentes')

    # Cria uma lista de todos os caminhos e seus labels
    file_paths = []
    
    # Adiciona imagens saudáveis (Label 0)
    for ext in ('.png', '.jpg', '.jpeg'):
        file_paths.extend([(p, 0) for p in glob(os.path.join(healthy_dir, f'*{ext}'))])
        
    # Adiciona imagens doentes (Label 1)
    for ext in ('.png', '.jpg', '.jpeg'):
        file_paths.extend([(p, 1) for p in glob(os.path.join(disease_dir, f'*{ext}'))])

    print(f"Total de {len(file_paths)} imagens encontradas para avaliação.")

    # 3. Loop de Avaliação
    for img_path, label in file_paths:
        
        # --- CARREGAMENTO E PRÉ-PROCESSAMENTO ---
        # Lê a imagem COLORIDA original
        import cv2 # cv2 precisa ser importado aqui se não for globalmente
        img_color_orig = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_color_orig is None:
             print(f"Aviso: Não foi possível ler a imagem {img_path}")
             continue
             
        img_color_orig = cv2.cvtColor(img_color_orig, cv2.COLOR_BGR2RGB)
        img_color_orig_256 = cv2.resize(img_color_orig, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)

        # Prepara PyTorch Tensors
        real_A, real_B = numpy_rgb_to_lab_tensor(img_color_orig_256)
        
        # --- INFERÊNCIA ---
        with torch.no_grad(): 
            model.set_input({'A': real_A, 'B': real_B, 'A_paths': [img_path]}) 
            model.forward() # Executa o forward pass G(A) -> fake_B

        # 4. Pós-processamento e Cálculo do Score
        # Chamamos o método interno do modelo para converter Lab -> RGB
        model.fake_B_rgb = model.lab2rgb(model.real_A, model.fake_B) 
        reconstructed_rgb_np = model.fake_B_rgb # (Convertido para NumPy 0-255 pelo método do modelo)
        
        # Cálculo do Score Final (CIEDE2000 - NumPy)
        score = calculate_anomaly_score_mean(img_color_orig_256, reconstructed_rgb_np)
        
        all_scores.append(score)
        all_labels.append(label)
        
        status = "DOENTE" if label == 1 else "SAUDÁVEL"
        print(f"Processado: {os.path.basename(img_path)} ({status}), Score CIEDE: {score:.4f}")

    # 5. Calcular Métricas Finais
    if len(all_labels) == 0:
        print("\nNenhuma imagem processada. Verifique os paths.")
        return [], []
        
    # Converte listas para arrays NumPy
    scores_np = np.array(all_scores)
    labels_np = np.array(all_labels)

    # Garante que há as duas classes para calcular as métricas
    if len(np.unique(labels_np)) < 2:
        print("\nAVISO: As métricas de classificação requerem pelo menos uma imagem saudável e uma doente.")
        return all_scores, all_labels
        
    # --- CÁLCULO DO AUC-ROC ---
    auc_roc = roc_auc_score(labels_np, scores_np)
    
    # --- CÁLCULO DO LIMIAR ÓPTIMO (Usando J de Youden) ---
    # roc_curve retorna FPR (Taxa de Falso Positivo), TPR (Taxa de Verdadeiro Positivo) e thresholds
    fpr, tpr, thresholds = roc_curve(labels_np, scores_np)
    
    # Youden's J statistic: Maximize J = TPR - FPR
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    # --- CLASSIFICAÇÃO BINÁRIA COM O LIMIAR ÓPTIMO ---
    # Se o score de anomalia for maior que o limiar, a previsão é 1 (DOENTE)
    predictions_np = (scores_np > optimal_threshold).astype(int)
    
    # --- CÁLCULO DAS MÉTRICAS DE CLASSIFICAÇÃO ---
    accuracy = accuracy_score(labels_np, predictions_np)
    precision = precision_score(labels_np, predictions_np, zero_division=0) # zero_division=0 para evitar avisos em caso de 0 previsões
    recall = recall_score(labels_np, predictions_np, zero_division=0)
    f1 = f1_score(labels_np, predictions_np, zero_division=0)
    
    print("\n" + "="*40)
    print("--- Resultados Finais ---")
    print(f"Número de amostras: {len(all_labels)}")
    print("="*40)
    print("\n--- Métrica Baseada em Score ---")
    print(f"AUC-ROC (CIEDE2000): {auc_roc:.4f}")
    
    print("\n--- Métricas Baseadas em Limiar ---")
    print(f"Limiar Óptimo (Youden's J): {optimal_threshold:.4f}")
    print(f"Acurácia (Accuracy): {accuracy:.4f}")
    print(f"Precisão (Precision): {precision:.4f}")
    print(f"Recall (Sensibilidade): {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("="*40)

    # Salva os scores para análise posterior
    np.save('anomaly_scores.npy', scores_np)
    np.save('anomaly_labels.npy', labels_np)
    np.save('optimal_predictions.npy', predictions_np)
    
    return all_scores, all_labels

if __name__ == '__main__':
    # Importação do cv2 aqui, pois é usado apenas no bloco __main__ e funções auxiliares
    import cv2
    run_metric_evaluation()