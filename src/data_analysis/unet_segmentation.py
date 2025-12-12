import torch
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T
import os

def load_model():
    """
    PyTorch Hubから学習済みのDeepLabV3モデルをロードします。
    """
    print("学習済みDeepLabV3モデルをロード中...")
    # 'pytorch/vision' からDeepLabV3 (ResNet101バックボーン) をロード
    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
    model.eval() # 評価モードに設定
    print("モデルのロードが完了しました。")
    return model

def segment_image(model, image_path):
    """
    指定された画像に対してセマンティックセグメンテーションを実行します。
    """
    try:
        input_image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"エラー: 画像ファイルが見つかりません: {image_path}")
        return None, None

    # モデルが学習されたときと同じ前処理を定義
    preprocess = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # バッチ次元を追加 (B, C, H, W)

    # GPUが利用可能ならGPUにテンソルを移動
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    # モデルで推論を実行
    with torch.no_grad():
        output = model(input_batch)['out'][0]
    
    # 各ピクセルに対して最もスコアの高いクラスを予測
    output_predictions = output.argmax(0)
    
    return input_image, output_predictions.cpu()

def visualize_segmentation(original_image, mask, save_path="./data/processed/images/segmentation_result.png"):
    """
    元の画像とセグメンテーションマスクを並べて表示・保存します。
    """
    # カラーパレットを作成（21クラス、Oxford-IIIT Pet Datasetのクラス数に合わせる）
    palette = torch.tensor([2**25 - 1, 2**15 - 1, 2**21 - 1])
    colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")

    # マスクをRGB画像に変換
    mask_rgb = Image.fromarray(mask.byte().numpy()).resize(original_image.size)
    mask_rgb.putpalette(colors)

    # Matplotlibで表示
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(mask_rgb)
    plt.title("Segmentation Mask")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"セグメンテーション結果を {save_path} に保存しました。")
    plt.show()

if __name__ == "__main__":
    # --- ユーザーが編集する部分 ---
    # セグメンテーションを行いたい画像ファイルのフルパスを指定してください
    IMAGE_PATH = "./data/raw/street_view_images_50m_optimized/pano_ZycTTS3CT78mb29T6fdfeQ_h180.jpg"
    # --- 編集はここまで ---

    if not os.path.isfile(IMAGE_PATH):
        print(f"エラー: 指定された画像パスにファイルが見つかりません: {IMAGE_PATH}")
    else:
        # 1. モデルのロード
        segmentation_model = load_model()

        # 2. セグメンテーションの実行
        print(f"\n画像 '{os.path.basename(IMAGE_PATH)}' のセグメンテーションを実行中...")
        original, segmentation_mask = segment_image(segmentation_model, IMAGE_PATH)
        
        # 3. 結果の可視化
        if original is not None:
            print("セグメンテーション結果を可視化中...")
            visualize_segmentation(original, segmentation_mask)
            print("\n処理が完了しました。")
