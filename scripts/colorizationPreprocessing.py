import cv2
import numpy as np
import os

# --- Configurações de Caminho ---
SOURCE_DIR = '../source_images/healthy_train_originals/'
TARGET_DIR = '../datasets/leaf_disease_detection/train/' 

# --- Parâmetros ---
REQUIRED_HEIGHT = 256
REQUIRED_WIDTH = 256

def preprocess_images(source_dir, target_dir):
    """
    Lê imagens coloridas, redimensiona para 256x256
    e salva no diretório de destino (SEM concatenar).
    """
    os.makedirs(target_dir, exist_ok=True)
    
    print(f"Buscando imagens em: {source_dir}")
    
    image_files = [
        f for f in os.listdir(source_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    
    if not image_files:
        print(f"ERRO: Nenhuma imagem encontrada em {source_dir}. Verifique o caminho.")
        return

    for i, filename in enumerate(image_files):
        source_path = os.path.join(source_dir, filename)

        # Lê imagem colorida
        img_color = cv2.imread(source_path, cv2.IMREAD_COLOR)

        if img_color is None:
            print(f"Aviso: Não foi possível ler a imagem: {filename}")
            continue

        # Redimensiona para 256x256
        img_resized = cv2.resize(
            img_color,
            (REQUIRED_WIDTH, REQUIRED_HEIGHT),
            interpolation=cv2.INTER_AREA
        )

        # Salva somente a imagem redimensionada (sem concatenação)
        target_path = os.path.join(target_dir, filename)
        cv2.imwrite(target_path, img_resized)

        print(f"Processado ({i+1}/{len(image_files)}): {filename} -> {target_path}")

    print("\n✅ Pré-processamento concluído! Imagens prontas para --model colorization.")

if __name__ == "__main__":
    preprocess_images(SOURCE_DIR, TARGET_DIR)
