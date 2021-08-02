import glob
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO
import xml.etree.ElementTree as ET


# 评价当前类簇中心(clusters)的优劣
def iou(box, clusters):
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    # or
    if np.count_nonzero(x == 0) > 0 and np.count_nonzero(y == 0) > 0:
        raise ValueError("Box has no area")
    intersection = x * y          # 交叉面积
    box_area = box[0] * box[1]    # 目标的面积
    cluster_area = clusters[:, 0] * clusters[:, 1]   # anchor面积
    iou_ = intersection / (box_area + cluster_area - intersection)  # 交并比
    return iou_


# 求类簇聚合程度
def avg_iou(boxes, clusters):
    return np.mean([np.max(iou(boxes[i], clusters)) for i in range(boxes.shape[0])])


# 返回kmeans对目标长宽信息的聚类信息
def Iou_Kmeans(boxes, k, dist=np.median):
    rows = boxes.shape[0]                 # 获取目标的数量
    distances = np.empty((rows, k))       # 中心点距离每个目标的距离
    last_clusters = np.zeros((rows,))     # 用做标志位，判断前后俩结果是否发生改变
    np.random.seed()
    clusters = boxes[np.random.choice(rows, k, replace=False)]  # 随机选择类簇中心点
    while True:
        for row in range(rows):
            distances[row] = 1 - iou(boxes[row], clusters)      # 计算iou距离，共中心点，比重叠率
        nearest_clusters = np.argmin(distances, axis=1)         # 对所有目标距离中心点排序
        if (last_clusters == nearest_clusters).all():
            break  # 前后两次未更新，则退出
        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)  # 求该类别下的目标，并更新中心点
        last_clusters = nearest_clusters
    return clusters


# coco数据集类别和id的映射
def id2name(coco):
    classes = dict()   # 构建类别与id的映射
    classes_id = []    # 类别id
    for cls in coco.dataset['categories']:
        classes[cls['id']] = cls['name']

    for key in classes.keys():
        classes_id.append(key)
    return classes, classes_id


# 获取所有目标的长宽数据
def load_dataset(path, types='voc'):
    dataset = []   # 用来记录每个目标的长宽信息
    if types == 'voc':
        # glob方法返回所有匹配后缀的文件
        for xml_file in glob.glob("{} /*xml".format(path)):
            tree = ET.parse(xml_file)
            height = int(tree.findtext("./size/height"))  # 图片长
            width = int(tree.findtext("./size/width"))    # 图片宽
            for obj in tree.iter("object"):
                xmin = int(obj.findtext("bndbox/xmin")) / width
                ymin = int(obj.findtext("bdbox/ymin")) / height
                xmax = int(obj.findtext("bndbox/xmax")) / width
                ymax = int(obj.findtext("bndbox/ymax")) / height
                xmin = np.float64(xmin)
                ymin = np.float64(ymin)
                xmax = np.float64(xmax)
                ymax = np.float64(ymax)
                if xmax == xmin or ymax == ymin:
                    print(xml_file)   # 输出有问题的目标
                dataset.append([xmax - xmin, ymax - ymin])  # 该目标相对图片的长宽
    if types == 'coco':
        coco = COCO(path)   # 构建coco类
        classes, classes_id = id2name(coco)  # coco类别数据和id的映射
        print(classes)      # 输出类别
        print('class_ids:', classes_id)      # 输出类别id
        img_ids = coco.getImgIds()  # 获取图像的id
        print(len(img_ids))         # 输出图像的id
        for imgId in img_ids:
            img = coco.loadImgs(imgId)[0]  # 获取对应id的图片信息
            height = img['height']  # 图片长
            width = img['width']    # 图片宽
            annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)  # 获取对应图片的标注信息
            anns = coco.loadAnns(annIds)   # 加载标注信息
            for ann in anns:
                if 'bbox' in ann:
                    bbox = ann['bbox']     # 获取目标的bbox
                    ann_width = bbox[2]    # 宽
                    ann_height = bbox[3]   # 长
                    ann_width = np.float64(ann_width / width)
                    ann_height = np.float64(ann_height / height)
                    dataset.append([ann_width, ann_height])         # 目标长宽信息
                else:
                    raise ValueError("coco no bbox -- wrong!!!")
    return np.array(dataset)


if __name__ == '__main__':
    annFile = 'train_all.json'     # coco格式的标注文件
    clusters = 12                                          # anchor数量（期望的长宽比例数）
    Inputdim = 800                                        # 理想的网络输入尺度  M*M
    data = load_dataset(path=annFile, types='coco')       # 加载coco标注信息，返回所有目标的长宽信息
    out = Iou_Kmeans(data, k=clusters)                    # kmeans聚类
    anchor = np.array(out) * Inputdim
    print("Boxes: {} ".format(anchor))                    #输出计算得到的anchor
    print("Accuracy: {:.2f}%".format(avg_iou(data, out) * 100))            # 计算正确率（评价anchor的代表性）
    final_anchors = np.around(out[:, 0] / out[:, 1], decimals=2).tolist()  # 输出长宽比
    print("Before Sort Ratios:\n {}".format(final_anchors))
    print("After Sort Ratios:\n {}".format(sorted(final_anchors)))