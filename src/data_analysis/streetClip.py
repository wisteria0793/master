# Load model directly
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
# 以下のライブラリを追記
import torch
from PIL import Image

# --- 元々のコード ---
processor = AutoProcessor.from_pretrained("geolocal/StreetCLIP")
model = AutoModelForZeroShotImageClassification.from_pretrained("geolocal/StreetCLIP")
# --------------------

# --- ここから追記 ---

# 1. 分類したい画像を開く
try:
    # ここに分類したい画像のパスを指定してください
    image_path = "data/raw/street_view_images_50m_optimized/pano_Z_vLZBn614K2XIsJUfla1g_h270.jpg"
    # image_path = "data/raw/street_view_images_50m_optimized/pano_0q-yXuD4A5hj4rSOXCjjEg_h270.jpg"
    # image_path = "data/raw/street_view_images_50m_optimized/pano_9UsWNHYoTVh1ZAmfwfZL9A_h0.jpg"   # 交差点
    # image_path = "data/raw/street_view_images_50m_optimized/pano_20evggJpR7-kn-V8LGj-PQ_h270.jpg"   # 駅前通り
    # image_path = "data/raw/street_view_images_50m_optimized/pano_3619beVjatq6BdKCBMLgkw_h180.jpg"   # 歩道
    # image_path = "data/raw/street_view_images_50m_optimized/pano_aeymQh7nNrxa9EQcTyRYNA_h0.jpg"   # 石畳

    image = Image.open(image_path).convert("RGB")
except FileNotFoundError:
    print(f"エラー: '{image_path}' が見つかりませんでした。画像パスを確認してください。")
    exit()

# 2. 分類に使いたいラベルのリストを定義する
# candidate_labels = ["urban area", "natural landscape", "residential street", "commercial district", "highway"]
# candidate_labels = ["asphalt road", "cobblestone road", "intersection", "gravel road", "paved road", "sidewalk", "trail", "Scattered hailstone pavement"]
# 路面の材質
candidate_labels = [
    # "asphalt road",      # アスファルト
    "concrete road",     # コンクリート（白っぽい舗装路）
    "cobblestone road",  # 石畳
    "brick road",        # レンガ道
    "gravel road",       # 砂利道
    "dirt road",         # 土の道（未舗装）
    "grassy trail"       # 草の生えた道
]
# 道路の種類・シーン
# candidate_labels = [
#     "highway",             # 高速道路
#     "urban street",        # 市街地の通り（ビルや店がある）
#     "residential street",  # 住宅街の道
#     "country road",        # 田舎道
#     "forest path",         # 森の小道
#     "alleyway",            # 路地・裏道
#     "parking lot"          # 駐車場（誤検知を防ぐための「その他」枠として有効）
# ]
# 構造
# candidate_labels = [
#     "straight road",    # 直線道路
#     "curved road",      # カーブ
#     "intersection",     # 交差点
#     "crosswalk",        # 横断歩道
#     "roundabout",       # 環状交差点
#     "sidewalk"          # 歩道
# ]


# 3. モデルへの入力を作成する
inputs = processor(images=image, text=candidate_labels, return_tensors="pt", padding=True)

# 4. モデルで分類を実行する
with torch.no_grad():
    outputs = model(**inputs)

# 5. 結果を取得して表示する
logits_per_image = outputs.logits_per_image
probs = logits_per_image.softmax(dim=1)  # スコアを確率に変換
scores = probs.squeeze().tolist()

print("--- 分類結果 ---")
for label, score in sorted(zip(candidate_labels, scores), key=lambda x: x[1], reverse=True):
    print(f"{label}: {score:.4f}")



# ベクトル抽出のみを行いたい場合のコード例
input = processor(images=image, return_tensors="pt")
image_features = model.get_image_features(pixel_values=input.pixel_values)

print(f"抽出された画像ベクトルの形状: {image_features.shape}")
# print(image_features)