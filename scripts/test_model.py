import sys
import os
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from skimage import color 
from glob import glob

# Importa as métricas
from sklearn.metrics import (
    roc_auc_score, 
    roc_curve, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix
)

import matplotlib
matplotlib.use('Agg')

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '..')
sys.path.append(project_root)

try:
    from models import create_model
    from options.test_options import TestOptions
except ImportError as e:
    print(f"ERRO CRÍTICO DE IMPORTAÇÃO: {e}")
    sys.exit(1)

# --- CONFIGURAÇÕES GLOBAIS ---
MODEL_NAME = 'final_model_15'
CHECKPOINTS_DIR = '../checkpoints'
DATASET_ROOT = '../datasets/leaf_disease_detection/test'
RESULTS_DIR = '../results/model_tests'
GLOBAL_REPORT_NAME = 'RELATORIO_GERAL_METRICAS_IMAGENS.pdf'
IMAGE_SIZE = 256
DEVICE = None 

# --- FUNÇÕES AUXILIARES ---

def setup_options():
    global DEVICE
    opt = TestOptions().parse()
    opt.name = MODEL_NAME
    opt.model = 'colorization'
    opt.dataset_mode = 'colorization'
    opt.netG = 'unet_256'
    opt.input_nc = 1 
    opt.output_nc = 2
    opt.norm = 'batch' 
    opt.epoch = '15' 
    opt.num_threads = 0
    opt.batch_size = 1
    opt.no_flip = True
    opt.checkpoints_dir = CHECKPOINTS_DIR
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

def numpy_rgb_to_lab_tensor(img_rgb_np):
    lab_np = color.rgb2lab(img_rgb_np / 255.0)
    lab_tensor = torch.from_numpy(lab_np.astype(np.float32)).to(DEVICE).permute(2, 0, 1)
    L_norm = lab_tensor[0:1, :, :] / 50.0 - 1.0
    ab_norm = lab_tensor[1:3, :, :] / 110.0
    return L_norm.unsqueeze(0), ab_norm.unsqueeze(0)

def process_anomaly(original_rgb_np, reconstructed_rgb_np):
    original_lab = color.rgb2lab(original_rgb_np / 255.0)
    reconstructed_lab = color.rgb2lab(reconstructed_rgb_np / 255.0)
    delta_e_map = color.deltaE_ciede2000(original_lab, reconstructed_lab)
    score_mean = np.mean(delta_e_map)
    return score_mean, delta_e_map

def plot_sample_row(ax_row, img_gray, img_orig, img_recon, diff_map, 
                   filename, score, pred_label, true_label, threshold, is_pdf_page=False):
    
    status_pred = "DOENTE" if pred_label == 1 else "SAUDÁVEL"
    status_real = "DOENTE" if true_label == 1 else "SAUDÁVEL"
    
    info_text = (f"Arq: {filename} | Score: {score:.4f}\n"
                 f"Pred: {status_pred} | Real: {status_real}")

    fontsize = 10 if is_pdf_page else 14

    ax_row[0].imshow(img_gray, cmap='gray')
    ax_row[0].axis('off')
    if not is_pdf_page: ax_row[0].set_title("Entrada (L)", fontsize=fontsize)

    ax_row[1].imshow(img_orig)
    ax_row[1].axis('off')
    if not is_pdf_page: ax_row[1].set_title("Original", fontsize=fontsize)

    ax_row[2].imshow(img_recon)
    ax_row[2].axis('off')
    if not is_pdf_page: ax_row[2].set_title("Reconstruída", fontsize=fontsize)

    im = ax_row[3].imshow(diff_map, cmap='jet', vmin=0, vmax=20)
    ax_row[3].axis('off')
    if not is_pdf_page: ax_row[3].set_title("Erro CIEDE", fontsize=fontsize)
    
    return info_text, im

# --- LOOP PRINCIPAL ---

def run_full_evaluation():
    opt = setup_options()
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print(f"Carregando modelo: {opt.name}...")
    model = create_model(opt)
    model.setup(opt) 
    model.eval()

    healthy_dir = os.path.join(DATASET_ROOT, 'saudaveis')
    disease_dir = os.path.join(DATASET_ROOT, 'doentes')
    
    file_paths = []
    for ext in ('.png', '.jpg', '.jpeg'):
        file_paths.extend([(p, 0) for p in glob(os.path.join(healthy_dir, f'*{ext}'))])
        file_paths.extend([(p, 1) for p in glob(os.path.join(disease_dir, f'*{ext}'))])
    
    print(f"Total de imagens: {len(file_paths)}")
    if len(file_paths) == 0: return

    # =================================================================
    # PASSO 1: CALCULAR SCORES E MÉTRICAS GLOBAIS
    # =================================================================
    print("\n--- PASSO 1: Calculando Estatísticas e Métricas ---")
    
    all_scores = []
    all_labels = []
    path_to_score = {}

    for i, (img_path, label) in enumerate(file_paths):
        img_bgr = cv2.imread(img_path)
        if img_bgr is None: continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb_256 = cv2.resize(img_rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        
        real_A, real_B = numpy_rgb_to_lab_tensor(img_rgb_256)
        
        with torch.no_grad():
            model.set_input({'A': real_A, 'B': real_B, 'A_paths': [img_path]})
            model.forward()
            reconstructed_rgb = model.lab2rgb(model.real_A, model.fake_B)
        
        score_mean, _ = process_anomaly(img_rgb_256, reconstructed_rgb)
        
        all_scores.append(score_mean)
        all_labels.append(label)
        path_to_score[img_path] = score_mean
        
        sys.stdout.write(f"\rProcessando: {i+1}/{len(file_paths)}")
        sys.stdout.flush()

    scores_np = np.array(all_scores)
    labels_np = np.array(all_labels)
    
    metrics_report = ""
    optimal_threshold = 0.0
    
    if len(np.unique(labels_np)) >= 2:
        fpr, tpr, thresholds = roc_curve(labels_np, scores_np)
        
        # --- NOVO CRITÉRIO: Otimizar para F1-Score ---
        f1_scores = []
        
        for t in thresholds:
            predictions = (scores_np > t).astype(int)
            # Precisão e Recall são necessários para o F1
            prec = precision_score(labels_np, predictions, zero_division=0)
            rec = recall_score(labels_np, predictions, zero_division=0)
            
            # F1-Score = 2 * (Precisão * Recall) / (Precisão + Recall)
            if (prec + rec) == 0:
                f1_scores.append(0)
            else:
                f1_scores.append(2 * (prec * rec) / (prec + rec))

        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        # Se você quisesse manter o limiar de Youden, usaria:
        # j_scores = tpr - fpr
        # optimal_idx = np.argmax(j_scores)
        # optimal_threshold = thresholds[optimal_idx]
        # ----------------------------------------------
        
        # Recalcular métricas com o NOVO limiar (F1-máximo)
        predictions_np = (scores_np > optimal_threshold).astype(int)
        
        auc = roc_auc_score(labels_np, scores_np) # AUC não muda
        acc = accuracy_score(labels_np, predictions_np)
        prec = precision_score(labels_np, predictions_np, zero_division=0)
        rec = recall_score(labels_np, predictions_np, zero_division=0)
        f1 = f1_score(labels_np, predictions_np, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(labels_np, predictions_np).ravel()

        metrics_txt = (
            f"========================================\n"
            f"  RELATÓRIO DE PERFORMANCE DO MODELO  \n"
            f"========================================\n"
            f"Modelo: {MODEL_NAME}\n"
            f"Total de Amostras: {len(labels_np)}\n"
            f"----------------------------------------\n"
            f"Limiar Otimizado (Max F1-Score): {optimal_threshold:.4f}\n" # Alteração
            f"AUC-ROC:              {auc:.4f}\n"
            f"----------------------------------------\n"
            f"Acurácia (Accuracy):  {acc:.4f}\n"
            f"Precisão (Precision): {prec:.4f}\n"
            f"Recall (Sensibilidade):{rec:.4f}\n"
            f"F1-Score:             {f1:.4f}\n"
            f"----------------------------------------\n"
            f"Matriz de Confusão:\n"
            f"TN: {tn} | FP: {fp}\n"
            f"FN: {fn} | TP: {tp}\n"
            f"========================================\n"
        )
    else:
        optimal_threshold = np.mean(scores_np)
        metrics_txt = "Aviso: Apenas uma classe disponível. Métricas não calculadas.\n"

    print("\n" + metrics_txt)
    
    with open(os.path.join(RESULTS_DIR, 'metrics_summary.txt'), 'w', encoding='utf-8') as f:
        f.write(metrics_txt)

    # =================================================================
    # PASSO 2: GERAR PDFS INDIVIDUAIS E PDF GLOBAL
    # =================================================================
    print("--- PASSO 2: Gerando Relatórios Visuais ---")
    
    global_pdf_path = os.path.join(RESULTS_DIR, GLOBAL_REPORT_NAME)
    
    batch_data = [] 
    SAMPLES_PER_PAGE = 3 
    
    with PdfPages(global_pdf_path) as global_pdf:
        
        # --- PÁGINA 1: CAPA COM MÉTRICAS ---
        fig_cover = plt.figure(figsize=(8.5, 11))
        fig_cover.text(0.1, 0.9, "Relatório Global de Testes", fontsize=20, weight='bold')
        fig_cover.text(0.1, 0.5, metrics_txt, fontsize=12, family='monospace', va='center')
        global_pdf.savefig(fig_cover)
        plt.close(fig_cover)
        
        for i, (img_path, label) in enumerate(file_paths):
            filename = os.path.basename(img_path)
            score = path_to_score.get(img_path, 0)
            prediction = 1 if score > optimal_threshold else 0
            
            # Carregamento e inferência
            img_bgr = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_rgb_256 = cv2.resize(img_rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
            
            real_A, real_B = numpy_rgb_to_lab_tensor(img_rgb_256)
            
            with torch.no_grad():
                model.set_input({'A': real_A, 'B': real_B, 'A_paths': [img_path]})
                model.forward()
                reconstructed_rgb = model.lab2rgb(model.real_A, model.fake_B)
            
            # L Channel para visualização
            L_tensor = model.real_A.squeeze().cpu().numpy()
            L_denorm = (L_tensor + 1.0) * 50.0 
            img_gray = (L_denorm * 2.55).astype(np.uint8)
            
            # Heatmap
            _, diff_map = process_anomaly(img_rgb_256, reconstructed_rgb)
            
            # --- A. SALVAR PDF INDIVIDUAL ---
            folder_category = "PRED_DOENTE" if prediction == 1 else "PRED_SAUDAVEL"
            save_folder = os.path.join(RESULTS_DIR, folder_category)
            os.makedirs(save_folder, exist_ok=True)
            
            fig_ind, axes_ind = plt.subplots(1, 4, figsize=(16, 5))
            # Ajuste individual
            plt.subplots_adjust(top=0.75, wspace=0.3) 
            
            info_txt, im_ind = plot_sample_row(axes_ind, img_gray, img_rgb_256, reconstructed_rgb.astype(np.uint8), diff_map,
                                       filename, score, prediction, label, optimal_threshold)
            
            fig_ind.suptitle(info_txt, fontsize=14, y=0.92, fontweight='bold')
            fig_ind.colorbar(im_ind, ax=axes_ind[3], fraction=0.046, pad=0.04)
            
            base_name = os.path.splitext(filename)[0]
            pdf_path = os.path.join(save_folder, f"{base_name}_score_{score:.2f}.pdf")
            fig_ind.savefig(pdf_path, format='pdf', dpi=100)
            plt.close(fig_ind)
            
            # --- B. PREPARAR DADOS PARA PDF GLOBAL ---
            batch_data.append({
                'gray': img_gray,
                'orig': img_rgb_256,
                'recon': reconstructed_rgb.astype(np.uint8),
                'diff': diff_map,
                'file': filename,
                'score': score,
                'pred': prediction,
                'label': label
            })
            
            # Se encheu a página do relatório global
            if len(batch_data) == SAMPLES_PER_PAGE:
                fig_batch, axes_batch = plt.subplots(SAMPLES_PER_PAGE, 4, figsize=(15, SAMPLES_PER_PAGE * 4))
                
                # --- AJUSTE DE LAYOUT GLOBAL (CORREÇÃO AQUI) ---
                # 'top' alterado de 0.92 para 0.85 para dar mais espaço (headroom) para a legenda da 1ª linha
                plt.subplots_adjust(top=0.85, bottom=0.05, hspace=0.8, wspace=0.3)
                
                for row_idx, data in enumerate(batch_data):
                    info, _ = plot_sample_row(axes_batch[row_idx], data['gray'], data['orig'], data['recon'], data['diff'],
                                           data['file'], data['score'], data['pred'], data['label'], optimal_threshold, is_pdf_page=True)
                    
                    # Legenda centralizada acima da linha
                    axes_batch[row_idx, 0].text(2.2, 1.35, info, 
                                                transform=axes_batch[row_idx, 0].transAxes, 
                                                ha='center', va='bottom', 
                                                fontsize=10, fontweight='bold')

                # Títulos das colunas 
                axes_batch[0, 0].set_title("Entrada (L)")
                axes_batch[0, 1].set_title("Original")
                axes_batch[0, 2].set_title("Reconstruída")
                axes_batch[0, 3].set_title("Mapa de Erro")
                
                global_pdf.savefig(fig_batch)
                plt.close(fig_batch)
                batch_data = []

            sys.stdout.write(f"\rGerando Relatórios: {i+1}/{len(file_paths)}")
            sys.stdout.flush()
        
        # --- C. PROCESSAR O RESTO DO BATCH (ÚLTIMA PÁGINA) ---
        if len(batch_data) > 0:
            fig_batch, axes_batch = plt.subplots(len(batch_data), 4, figsize=(15, len(batch_data) * 4))
            if len(batch_data) == 1: axes_batch = np.expand_dims(axes_batch, axis=0)
            
            # Mesmo ajuste de top=0.85 aqui também
            plt.subplots_adjust(top=0.85, bottom=0.05, hspace=0.8, wspace=0.3)
            
            for row_idx, data in enumerate(batch_data):
                info, _ = plot_sample_row(axes_batch[row_idx], data['gray'], data['orig'], data['recon'], data['diff'],
                                       data['file'], data['score'], data['pred'], data['label'], optimal_threshold, is_pdf_page=True)
                
                axes_batch[row_idx, 0].text(2.2, 1.35, info, 
                                            transform=axes_batch[row_idx, 0].transAxes, 
                                            ha='center', va='bottom', 
                                            fontsize=10, fontweight='bold')
            
            axes_batch[0, 0].set_title("Entrada (L)")
            axes_batch[0, 1].set_title("Original")
            axes_batch[0, 2].set_title("Reconstruída")
            axes_batch[0, 3].set_title("Mapa de Erro")
            
            global_pdf.savefig(fig_batch)
            plt.close(fig_batch)

    print(f"\n\nConcluído!")
    print(f"Relatório Global: {os.path.abspath(global_pdf_path)}")
    print(f"PDFs Individuais salvos em: {os.path.abspath(RESULTS_DIR)}")

if __name__ == '__main__':
    run_full_evaluation()