# jsonTOxml.py
# 用于将labelme标注的json文件转换为labelimg的VOC格式的xml文件


import os
import json
import xml.etree.ElementTree as ET
from xml.dom import minidom

def json_to_xml(json_path, xml_dir):
    # 读取json文件
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
  
    # 创建XML结构
    annotation = ET.Element('annotation')
  
    # 添加文件夹信息
    folder = ET.SubElement(annotation, 'folder')
    folder.text = 'images'
  
    # 添加文件名
    filename = ET.SubElement(annotation, 'filename')
    filename.text = data['imagePath']
  
    # 添加路径
    path = ET.SubElement(annotation, 'path')
    path.text = os.path.abspath(os.path.join(xml_dir, data['imagePath']))
  
    # 添加来源信息
    source = ET.SubElement(annotation, 'source')
    database = ET.SubElement(source, 'database')
    database.text = 'Unknown'
  
    # 添加图片尺寸
    size = ET.SubElement(annotation, 'size')
    width = ET.SubElement(size, 'width')
    width.text = str(data['imageWidth'])
    height = ET.SubElement(size, 'height')
    height.text = str(data['imageHeight'])
    depth = ET.SubElement(size, 'depth')
    depth.text = '3'
  
    # 添加分割信息
    segmented = ET.SubElement(annotation, 'segmented')
    segmented.text = '0'
  
    # 处理每个标注形状
    for shape in data['shapes']:
        obj = ET.SubElement(annotation, 'object')
  
        # 添加对象名称
        name = ET.SubElement(obj, 'name')
        name.text = shape['label']
  
        # 添加姿态信息
        pose = ET.SubElement(obj, 'pose')
        pose.text = 'Unspecified'
  
        # 添加截断信息
        truncated = ET.SubElement(obj, 'truncated')
        truncated.text = '0'
  
        # 添加难度信息
        difficult = ET.SubElement(obj, 'difficult')
        difficult.text = '0'
  
        # 添加边界框
        bndbox = ET.SubElement(obj, 'bndbox')
  
        # 计算边界框坐标
        points = shape['points']
        xmin = ET.SubElement(bndbox, 'xmin')
        xmin.text = str(int(min(p[0] for p in points)))
        ymin = ET.SubElement(bndbox, 'ymin')
        ymin.text = str(int(min(p[1] for p in points)))
        xmax = ET.SubElement(bndbox, 'xmax')
        xmax.text = str(int(max(p[0] for p in points)))
        ymax = ET.SubElement(bndbox, 'ymax')
        ymax.text = str(int(max(p[1] for p in points)))
  
    # 创建ElementTree对象
    tree = ET.ElementTree(annotation)
  
    # 写入XML文件（不包含XML声明）
    xml_filename = os.path.splitext(data['imagePath'])[0] + '.xml'
    xml_path = os.path.join(xml_dir, xml_filename)
  
    # 设置缩进并写入文件
    ET.indent(tree, space="  ")
    tree.write(xml_path, encoding='utf-8', xml_declaration=False)

def convert_all_json(json_dir, xml_dir):
    # 确保输出目录存在
    os.makedirs(xml_dir, exist_ok=True)
  
    # 遍历json目录下所有.json文件
    for filename in os.listdir(json_dir):
        if filename.endswith('.json'):
            json_path = os.path.join(json_dir, filename)
            try:
                json_to_xml(json_path, xml_dir)
                print(f'Converted {filename} successfully')
            except Exception as e:
                print(f'Failed to convert {filename}: {str(e)}')

if __name__ == '__main__':
    # 设置输入输出目录
    input_dir = 'json'
    output_dir = 'xml'
  
    # 执行转换
    convert_all_json(input_dir, output_dir)
