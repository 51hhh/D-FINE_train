# watch.py
# 查看标注框是否正确，检查图片和XML匹配情况
import cv2
import os
import xml.etree.ElementTree as ET


# 目录配置
script_dir = os.path.dirname(os.path.abspath(__file__))
img_dir = os.path.join(script_dir, 'images')
xml_dir = os.path.join(script_dir, 'Annotations')

# 显示尺寸配置
screen_width = 640
screen_height = 640


def get_base_names(directory, extension):
    """获取目录下所有指定扩展名文件的基础名（不含扩展名）"""
    files = []
    for f in os.listdir(directory):
        if f.lower().endswith(extension) and not f.startswith('.'):
            files.append(os.path.splitext(f)[0])
    return set(files)


def parse_xml(xml_path):
    """解析XML文件，返回边界框列表"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        boxes = []
        for obj in root.findall('object'):
            bndbox = obj.find('bndbox')
            if bndbox is not None:
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)
                name = obj.find('name').text if obj.find('name') is not None else 'unknown'
                boxes.append({'name': name, 'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax})
        return boxes
    except Exception as e:
        print(f"解析XML出错 {xml_path}: {e}")
        return []


def check_matching():
    """检查图片和XML的匹配情况，返回分类后的图片列表"""
    print("\n" + "=" * 50)
    print("检查图片和XML匹配情况")
    print("=" * 50)
    
    # 获取所有图片和XML的基础名
    image_names = get_base_names(img_dir, ('.png', '.jpg'))
    xml_names = get_base_names(xml_dir, '.xml')
    
    # 找出没有匹配上的
    images_without_xml = image_names - xml_names
    xml_without_images = xml_names - image_names
    matched = image_names & xml_names
    
    print(f"\n总图片数: {len(image_names)}")
    print(f"总XML数: {len(xml_names)}")
    print(f"已匹配数: {len(matched)}")
    
    # 输出没有XML的图片
    if images_without_xml:
        print(f"\n[警告] {len(images_without_xml)} 张图片没有对应的XML:")
        for name in sorted(images_without_xml)[:20]:  # 最多显示20个
            print(f"  - {name}.jpg")
        if len(images_without_xml) > 20:
            print(f"  ... 还有 {len(images_without_xml) - 20} 个")
    else:
        print("\n[OK] 所有图片都有对应的XML")
    
    # 输出没有图片的XML
    if xml_without_images:
        print(f"\n[警告] {len(xml_without_images)} 个XML没有对应的图片:")
        for name in sorted(xml_without_images)[:20]:
            print(f"  - {name}.xml")
        if len(xml_without_images) > 20:
            print(f"  ... 还有 {len(xml_without_images) - 20} 个")
    else:
        print("\n[OK] 所有XML都有对应的图片")
    
    # 检查没有检测框的XML
    print("\n" + "-" * 50)
    print("检查XML中没有检测框的文件")
    print("-" * 50)
    
    empty_xml = []
    normal_matched = []
    for name in matched:
        xml_path = os.path.join(xml_dir, name + '.xml')
        boxes = parse_xml(xml_path)
        if len(boxes) == 0:
            empty_xml.append(name)
        else:
            normal_matched.append(name)
    
    if empty_xml:
        print(f"\n[警告] {len(empty_xml)} 个XML没有检测框:")
        for name in sorted(empty_xml)[:20]:
            print(f"  - {name}.xml")
        if len(empty_xml) > 20:
            print(f"  ... 还有 {len(empty_xml) - 20} 个")
    else:
        print("\n[OK] 所有XML都包含检测框")
    
    print("\n" + "=" * 50 + "\n")
    
    # 构建优先级列表：(name, status)
    # status: 'no_xml' - 无XML, 'no_box' - 无检测框, 'normal' - 正常
    priority_list = []
    
    # 1. 优先：没有XML的图片
    for name in sorted(images_without_xml):
        priority_list.append((name, 'no_xml'))
    
    # 2. 其次：没有检测框的图片
    for name in sorted(empty_xml):
        priority_list.append((name, 'no_box'))
    
    # 3. 最后：正常匹配的图片
    for name in sorted(normal_matched):
        priority_list.append((name, 'normal'))
    
    return priority_list


def preview_images(priority_list):
    """预览图片，使用a/d切换，q退出
    priority_list: [(name, status), ...] status可为 'no_xml', 'no_box', 'normal'
    """
    if not priority_list:
        print("没有可预览的图片")
        return
    
    print("\n预览模式:")
    print("  a - 上一张")
    print("  d - 下一张") 
    print("  q - 退出")
    print("-" * 30)
    
    # 状态显示配置
    status_config = {
        'no_xml': {'text': 'NO XML', 'color': (0, 0, 255)},      # 红色
        'no_box': {'text': 'NO BOX', 'color': (0, 165, 255)},    # 橙色
        'normal': {'text': 'OK', 'color': (0, 255, 0)}           # 绿色
    }
    
    current_idx = 0
    window_name = "Image Preview"
    
    while True:
        name, status = priority_list[current_idx]
        img_path = os.path.join(img_dir, name + '.jpg')
        xml_path = os.path.join(xml_dir, name + '.xml')
        
        # 读取图片
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图片: {img_path}")
            current_idx = (current_idx + 1) % len(priority_list)
            continue
        
        # 缩放图片
        scale = min(screen_width / img.shape[1], screen_height / img.shape[0])
        new_size = (int(img.shape[1] * scale), int(img.shape[0] * scale))
        img = cv2.resize(img, new_size)
        
        # 如果有XML，解析并绘制边界框
        if status != 'no_xml' and os.path.exists(xml_path):
            boxes = parse_xml(xml_path)
            for box in boxes:
                x1 = int(box['xmin'] * scale)
                y1 = int(box['ymin'] * scale)
                x2 = int(box['xmax'] * scale)
                y2 = int(box['ymax'] * scale)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, box['name'], (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 左上角：显示当前索引信息
        info_text = f"[{current_idx + 1}/{len(priority_list)}] {name}"
        cv2.putText(img, info_text, (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # 右上角：显示状态
        status_info = status_config.get(status, status_config['normal'])
        status_text = status_info['text']
        status_color = status_info['color']
        
        # 计算文字位置（右上角）
        (text_width, text_height), _ = cv2.getTextSize(
            status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        text_x = new_size[0] - text_width - 10
        text_y = 30
        
        # 绘制背景矩形使文字更清晰
        cv2.rectangle(img, (text_x - 5, text_y - text_height - 5), 
                     (text_x + text_width + 5, text_y + 5), (0, 0, 0), -1)
        cv2.putText(img, status_text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        # 显示图片
        cv2.imshow(window_name, img)
        
        # 等待按键
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('a'):
            current_idx = (current_idx - 1) % len(priority_list)
        elif key == ord('d'):
            current_idx = (current_idx + 1) % len(priority_list)
    
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # 检查匹配情况并获取分类后的图片列表
    priority_list = check_matching()
    
    # 统计各类型数量
    no_xml_count = sum(1 for _, s in priority_list if s == 'no_xml')
    no_box_count = sum(1 for _, s in priority_list if s == 'no_box')
    normal_count = sum(1 for _, s in priority_list if s == 'normal')
    
    # 询问是否预览
    if priority_list:
        print(f"预览顺序: 无XML({no_xml_count}) -> 无检测框({no_box_count}) -> 正常({normal_count})")
        response = input(f"\n共 {len(priority_list)} 张图片，是否预览? (y/n): ")
        if response.lower() == 'y':
            preview_images(priority_list)