"""
@file   : preprocess_data_and_concat.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-06-18$
"""
import json
import os
from tqdm import tqdm
import xml.etree.ElementTree as ET  # 解析xml文件所用工具


def parse_annotation(annotation_path):
    # 解析xml文件，最终返回这张图片中所有目标的标注框及其类别信息，以及这个目标是否是一个difficult目标
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    boxes, labels, difficulties = [], [], []  # 存储bbox   bbox对应的label  bbox对应的difficult信息
    for object in root.iter('object'):
        difficult = int(object.find('difficult').text == '1')
        label = object.find('name').text.lower().strip()

        if label not in label_map:
            continue
        bbox = object.find('bndbox')
        xmin = int(bbox.find('xmin').text) - 1
        ymin = int(bbox.find('ymin').text) - 1
        xmax = int(bbox.find('xmax').text) - 1
        ymax = int(bbox.find('ymax').text) - 1

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label_map[label])
        difficulties.append(difficult)
    # 返回包含图片标注信息的字典
    return {'boxes': boxes, 'labels': labels, 'difficulties': difficulties}


def create_data_lists(voc07_path, voc12_path, output_folder):
    voc07_path = os.path.abspath(voc07_path)   # 获取绝对路径
    voc12_path = os.path.abspath(voc12_path)

    train_images, train_objects = [], []
    n_objects = 0

    # 训练数据
    for path in [voc07_path, voc12_path]:
        # 获取训练和验证图片id
        with open(os.path.join(path, "ImageSets/Main/trainval.txt")) as f:
            ids = f.read().splitlines()

        # 根据图片id 解析图片的xml文件 获取标注信息
        for id in tqdm(ids):
            objects = parse_annotation(os.path.join(path, 'Annotations', id + '.xml'))
            if len(objects['boxes']) == 0:
                # 如果没有目标  则跳过
                continue

            n_objects += len(objects['boxes'])   # 统计目标总数
            train_objects.append(objects)   # 存储每张图片的标注信息到列表train_objects
            train_images.append(os.path.join(path, 'JPEGImages', id + '.jpg'))  # 存储每张图片的位置
    assert len(train_objects) == len(train_images)

    # 将训练数据的图片路径，标注信息，类别映射信息，分别保存为json文件
    with open(os.path.join(output_folder, 'TRAIN_images.json'), 'w') as j:
        json.dump(train_images, j)

    with open(os.path.join(output_folder, 'TRAIN_objects.json'), 'w') as j:
        json.dump(train_objects, j)

    with open(os.path.join(output_folder, 'label_map.json'), 'w') as j:
        json.dump(label_map, j)

    print('训练集:{}, 目标数:{}, 处理完的数据保存到:{}'.format(
        len(train_images), n_objects, os.path.abspath(output_folder)
    ))

    # 对test数据处理。目的是将测试数据的图片路径，标注信息，类别映射信息，分别保存为json文件，参考上面的注释理解
    test_images, test_objects = [], []
    n_objects = 0
    with open(os.path.join(voc07_path, 'ImageSets/Main/test.txt')) as f:
        ids = f.read().splitlines()

    for id in tqdm(ids):
        objects = parse_annotation(os.path.join(voc07_path, 'Annotations', id + '.xml'))
        if len(objects['boxes']) == 0:
            continue
        test_objects.append(objects)
        n_objects += len(objects['boxes'])
        test_images.append(os.path.join(voc07_path, 'JPEGImages', id + '.jpg'))

    assert len(test_objects) == len(test_images)

    with open(os.path.join(output_folder, 'TEST_images.json'), 'w') as j:
        json.dump(test_images, j)
    with open(os.path.join(output_folder, 'TEST_objects.json'), 'w') as j:
        json.dump(test_objects, j)

    print('测试集:{}, 目标数:{}, 处理完的数据保存到:{}'.format(
        len(test_images), n_objects, os.path.abspath(output_folder)
    ))


if __name__ == '__main__':
    # Label map
    # voc_labels为VOC数据集中20类目标的类别名称
    voc_labels = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                  'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

    # 创建label_map字典，用于存储类别和类别索引之间的映射关系。比如：{1：'aeroplane'， 2：'bicycle'，......}
    label_map = {k: v + 1 for v, k in enumerate(voc_labels)}
    # VOC数据集默认不含有20类目标中的其中一类的图片的类别为background，类别索引设置为0
    label_map['background'] = 0

    # 将映射关系倒过来，{类别名称：类别索引}
    rev_label_map = {v: k for k, v in label_map.items()}  # Inverse mapping

    create_data_lists(voc07_path='../../input_data/VOCdevkit/VOC2007',
                      voc12_path='../../input_data/VOCdevkit/VOC2012',
                      output_folder='../../input_data/VOCdevkit')


