import cv2
import numpy as np
import os

# --- Configurações de Caminho ---
# PASTA ONDE VOCÊ COLOCOU TEMPORARIAMENTE AS 50 IMAGENS SAUDÁVEIS ORIGINAIS
# Certifique-se de que este caminho está correto no seu sistema!
SOURCE_DIR = '../source_images/healthy_train_originals/'

# PASTA DE DESTINO ESPERADA PELO PIX2PIX
# Onde as imagens concatenadas serão salvas (datasets/folhas_anomalia/train)
TARGET_DIR = '../datasets/leaf_desease_detection/train/' 

# --- Parâmetros ---
# O pix2pix espera que A e B tenham a mesma altura
REQUIRED_HEIGHT = 256
REQUIRED_WIDTH = 256 # A largura original será 2*WIDTH, mas cada imagem (A ou B) terá WIDTH

# ----------------------------------------------------

def generate_concat_pair(source_dir, target_dir):
    """
    Lê imagens coloridas, gera a versão em escala de cinza e concatena
    [Cinza | Cor] lado a lado.
    """
    # 1. Cria o diretório de destino se ele não existir
    os.makedirs(target_dir, exist_ok=True)
    
    print(f"Buscando imagens em: {source_dir}")
    
    # 2. Lista todos os arquivos de imagem na pasta de origem
    image_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"ERRO: Nenhuma imagem encontrada em {source_dir}. Verifique o caminho.")
        return

    for i, filename in enumerate(image_files):
        source_path = os.path.join(source_dir, filename)
        
        # 3. Lê a imagem (BGR colorida, que é o padrão do OpenCV)
        img_color = cv2.imread(source_path, cv2.IMREAD_COLOR)

        if img_color is None:
            print(f"Aviso: Não foi possível ler a imagem: {filename}")
            continue

        # 4. Redimensionamento para o tamanho padrão do pix2pix
        # Este passo é crucial, pois o pix2pix original trabalha com 256x256
        img_color = cv2.resize(img_color, (REQUIRED_WIDTH, REQUIRED_HEIGHT), interpolation=cv2.INTER_AREA)

        # 5. Gera a Imagem A (Escala de Cinza)
        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        # Converte a imagem cinza (1 canal) de volta para BGR (3 canais) para concatenação
        img_gray_3ch = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

        # 6. Concatena horizontalmente: [Cinza | Cor]
        # O lado esquerdo (A) é a entrada (Cinza), o lado direito (B) é a saída (Cor)
        img_concatenated = np.hstack([img_gray_3ch, img_color])

        # 7. Salva a imagem concatenada no diretório de destino
        target_path = os.path.join(target_dir, filename)
        cv2.imwrite(target_path, img_concatenated)
        
        print(f"Processado ({i+1}/{len(image_files)}): {filename} -> Salvo em {target_path}")

    print("\n✅ Pré-processamento concluído!")

if __name__ == "__main__":
    generate_concat_pair(SOURCE_DIR, TARGET_DIR)