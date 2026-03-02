"""为困难负样本创建 COCO 格式标注（无目标图片）"""
import json
from pathlib import Path
from PIL import Image
from datetime import datetime

# 配置
negative_img_dir = Path("coco/images/negative_samples")  # 误检图片目录
output_file = Path("coco/converted/annotations/hard_negatives.json")

# 检查目录
if not negative_img_dir.exists():
    print(f"请先创建目录并放入误检图片: {negative_img_dir}")
    exit(1)

# 加载现有训练集
train_file = Path("coco/converted/annotations/train.json")
train_data = json.load(open(train_file))

# 获取当前最大ID
max_img_id = max([img["id"] for img in train_data["images"]])

# 扫描负样本图片
negative_images = []
img_id = max_img_id + 1

for img_path in negative_img_dir.glob("*.jpg"):
    try:
        img = Image.open(img_path)
        width, height = img.size
        
        negative_images.append({
            "id": img_id,
            "file_name": f"negative_samples/{img_path.name}",
            "width": width,
            "height": height,
            "date_captured": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
        img_id += 1
        print(f"添加负样本: {img_path.name} ({width}x{height})")
    except Exception as e:
        print(f"跳过 {img_path.name}: {e}")

if len(negative_images) == 0:
    print(f"\n未找到负样本图片，请将误检的图片放到: {negative_img_dir}")
    exit(1)

print(f"\n找到 {len(negative_images)} 张负样本图片")

# 合并到训练集
new_train_data = {
    "images": train_data["images"] + negative_images,
    "annotations": train_data["annotations"],  # 负样本没有标注
    "categories": train_data["categories"],
}

# 保存增强训练集
output_train = Path("coco/converted/annotations/train_with_negatives.json")
with open(output_train, 'w', encoding='utf-8') as f:
    json.dump(new_train_data, f, indent=2, ensure_ascii=False)

print(f"\n✅ 已生成增强训练集: {output_train}")
print(f"   原始图片: {len(train_data['images'])} 张")
print(f"   负样本: {len(negative_images)} 张")
print(f"   总计: {len(new_train_data['images'])} 张")
print(f"\n下一步：更新配置文件使用新标注文件")
