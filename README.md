# 🌱 Diagnóstico de Doenças em Folhas usando IA Generativa  
### Projeto 2 — Introdução à Inteligência Artificial (UnB, 2025/2)

**Professor:** Díbio L. Borges 
**Departamento de Ciência da Computação — Universidade de Brasília**

---

## 📌 Visão Geral

Este repositório contém a implementação completa do **Projeto 2**, cujo objetivo é desenvolver um sistema de detecção de anomalias em folhas de plantas utilizando um método **não supervisionado** baseado em **reconstrução de cores**, conforme descrito no artigo:

**KATAFUCHI, Ryoya; TOKUNAGA, Terumasa.**  
*Image-based Plant Disease Diagnosis with Unsupervised Anomaly Detection Based on Reconstructability of Colors.*  
arXiv preprint arXiv:2011.14306, 2020.  
🔗 **PDF:** https://arxiv.org/pdf/2011.14306  

O método utiliza um modelo **pix2pix (GAN condicional)** para reconstruir imagens coloridas de folhas saudáveis a partir de suas versões em tons de cinza. Em seguida, calcula-se um índice perceptual de anomalia usando **CIEDE2000**, indicando possíveis regiões sintomáticas.

Além disso, o projeto inclui **visualização por Grad-CAM**, conforme:

**SELVARAJU, R.R. et al.**  
*Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization.*  
ICCV, 2017.  

---

## 🚀 Objetivos do Projeto

- Treinar o modelo **pix2pix** usando **50 imagens saudáveis**.
- Reconstruir cores de imagens saudáveis e doentes.
- Calcular anomalia usando **CIEDE2000** por pixel.
- Gerar mapas de calor das regiões sintomáticas.
- Avaliar o modelo seguindo as métricas do artigo:
  - ✔️ AUC-ROC   
  - ✔️ Acurácia
  - ✔️ Precisão  
  - ✔️ Recall  
  - ✔️ F1-score  
- Aplicar **Grad-CAM** ao modelo para identificar regiões relevantes.

---

## 🧠 Metodologia Utilizada

### **1. Reconstrução de Cores com pix2pix**
Configuração e hiperparâmetros

- **Gerador:** U-Net com *skip connections*  
- **Discriminador:** PatchGAN 70×70 
- **Loss:** GAN + L1 (λ = 5)  
- **Otimizador:** Adam  
  - lr = 0.00015  
  - β1 = 0.5  

### **2. Detecção de Anomalias com CIEDE2000**
- Para cada pixel:
  - Diferença perceptual entre imagem original e reconstruída
- Soma dos valores gera o **índice de anomalia**:
  - 🔴 Alto → possível região doente  
  - 🟢 Baixo → região saudável  

### **3. Visualização com Grad-CAM**
Aplicada ao modelo para destacar regiões decisivas para as previsões.

---

## 📊 Métricas (mesmos moldes do artigo)

- **AUC-ROC**
- **Acurácia**
- **Precisão**
- **Recall**
- **F1-score**
- **Histogramas do índice de anomalia**

As métricas utilizam:

- 50 imagens saudáveis (teste)
- 100 imagens doentes (teste)

---

## 🛠️ Como Executar

Para garantir que todas as dependências do PyTorch e pacotes de visualização sejam instaladas corretamente, é **fortemente recomendado** utilizar um ambiente virtual, como **Conda** ou **Miniconda**.

### 1. 🐍 Configurar o Ambiente Conda

Instale o Conda/Miniconda (se ainda não tiver) e crie o ambiente usando o arquivo de especificação fornecido:

```bash
# Crie o ambiente Conda a partir do arquivo environment.yml
conda env create -f environment.yml

# Ative o novo ambiente
conda activate <nome_do_seu_ambiente>
```

### 2. 📁 Estrutura de Dados

Certifique-se de que seu dataset esteja organizado no formato **pix2pix** no diretório `datasets/leaf_disease_detection`.

### 3. 🚀 Executar os Scripts

Os principais *scripts* de inferência e visualização estão localizados na pasta `scripts/`. O caminho para a raiz dos dados (`--dataroot`) deve ser especificado:

#### A. Testar o Modelo em Todo o Conjunto de Imagens

Executa a inferência completa, reconstruindo todas as imagens do conjunto de teste, calculando o CIEDE2000 e salvando os resultados em results/.

```bash
python scripts/test_model.py --dataroot ./datasets/leaf_disease_detection
```

#### B. Testar o Modelo em Uma Única Imagem

Permite verificar a reconstrução e o mapa CIEDE2000 de uma imagem específica — útil para análise qualitativa.

```bash
python scripts/test_single_image.py
--dataroot ./datasets/leaf_disease_detection
--path "../datasets/leaf_disease_detection/test/doentes/a988-992_ab_0.jpg"
```

#### C. Gerar Visualizações Grad-CAM

Gera visualizações Grad-CAM das camadas convolucionais do discriminador, permitindo interpretar quais regiões influenciam sua decisão.

```bash
python scripts/show_GradCAM.py --dataroot ./datasets/leaf_disease_detection
```

#### D. Treinar o Modelo Pix2Pix

Executa o processo completo de treinamento, salvando os checkpoints em checkpoints/.

```bash
python train.py
--dataroot D:/pytorch-CycleGAN-and-pix2pix/datasets/leaf_disease_detection
--name pix2pix_final_v3
--model colorization
--dataset_mode colorization
--direction AtoB
--lr 0.00015
--lambda_L1 5.0
--beta1 0.5
--n_epochs 100
--n_epochs_decay 50
--netG unet_256
--netD basic
--num_threads 0
```

---

# 🎨 Exemplos de Visualização

### 🔹 Reconstrução via pix2pix

| Original (Preto e Branco) | Original (RGB) | Reconstruída (RGB) | Mapa CIEDE2000 |
| :---: | :---: | :---: | :---: |
| ![Imagem em tons de cinza](results/test_single_leaf/reconstruction_a988-992_ab_0/input_grayscale_256.png) | ![Imagem RGB original](results/test_single_leaf/reconstruction_a988-992_ab_0/original_rgb_256.png) | ![Imagem RGB reconstruída](results/test_single_leaf/reconstruction_a988-992_ab_0/reconstructed_rgb_256.png) | ![Mapa CIEDE2000](results/test_single_leaf/reconstruction_a988-992_ab_0/ciede_heatmap_256.png) |
| **Localização:** `../results/.../input_grayscale_256.png` | **Localização:** `../results/.../original_rgb_256.png` | **Localização:** `../results/.../reconstructed_rgb_256.png` | **Localização:** `../results/.../ciede_heatmap_256.png` |



---

### 🔹 Grad-CAM

| Mapa de Calor Grad-CAM |
| :---: |
| ![Mapa de Calor Grad-CAM](results/Grad-CAM_layers/SAUDAVEIS/imagens/gradcam_leaf%20a1-a3%20ab_0_jpg.png) |
| **Localização:** `results/Grad-CAM_layers/...` |

---


# 👥 Autores

### 🔹 Pedro Marcinoni 

### 🔹 Leonardo Krauss 

### Projeto desenvolvido para a disciplina Introdução à Inteligência Artificial (CIC/UnB) — 2025/2.

---

# 📚 Referências

KATAFUCHI, R.; TOKUNAGA, T.
Image-based Plant Disease Diagnosis with Unsupervised Anomaly Detection Based on Reconstructability of Colors.
🔗 https://arxiv.org/pdf/2011.14306

SELVARAJU, R. et al.
Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization.

# 🙏 Créditos e Agradecimentos

Este projeto utiliza partes substanciais da implementação oficial de **pix2pix** disponibilizada pelo repositório:

### PyTorch CycleGAN and pix2pix

por **Jun-Yan Zhu**, **Taesung Park**, apoiado por **Tongzhou Wang**.

🔗 **Repositório:** [https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

O código completo fornece implementações em PyTorch para **CycleGAN** e **pix2pix**, compatíveis com o artigo:

1.  **Image-to-Image Translation with Conditional Adversarial Networks**
    * **Autores:** Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, Alexei A. Efros
    * **Conferência:** CVPR 2017.

e também:

2.  **Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks**
    * **Autores:** Jun-Yan Zhu, Taesung Park, Phillip Isola, Alexei A. Efros
    * **Conferência:** ICCV 2017.

### 📝 Como Citar (BibTeX)

Se você utilizar este trabalho academicamente, considere também citar os autores originais conforme instruído no repositório:

* **CycleGAN BibTeX:** [https://junyanz.github.io/CycleGAN/CycleGAN.txt](https://junyanz.github.io/CycleGAN/CycleGAN.txt)
* **pix2pix BibTeX:** [https://www.cs.cmu.edu/~junyanz/projects/pix2pix/pix2pix.bib](https://www.cs.cmu.edu/~junyanz/projects/pix2pix/pix2pix.bib)

A equipe de autores mantém documentação útil, guias de treinamento e notebooks educacionais que auxiliaram no desenvolvimento deste projeto.