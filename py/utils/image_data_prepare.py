import cv2
import os
import xmltodict
import pandas as pd

prefix_dir = "C:\\Users\\wangheng\\Documents\\software_cup\\train_card_images\\"
prefix_xml_dir = "C:\\Users\\wangheng\\Documents\\software_cup\\card_image_labs\\"
image_files = os.listdir(prefix_dir)
xml_files = os.listdir(prefix_xml_dir)


def change_size():
    for file in image_files:
        image = cv2.imread(prefix_dir + file, cv2.IMREAD_ANYCOLOR)
        resize = cv2.resize(image, (520, 390))
        cv2.imwrite(prefix_dir + file, resize)


def xml_to_csv_file():
    datas = []
    for xml_file in xml_files:
        with open(prefix_xml_dir + xml_file) as f:
            doc = xmltodict.parse(f.read())
            objects = doc['annotation']['object']
            for obj in objects:
                dict_obj = {'xmin': obj['bndbox']['xmin'], 'ymin': obj['bndbox']['ymin'],
                            'xmax': obj['bndbox']['xmax'], 'ymax':obj['bndbox']['ymax'],
                            'label':obj['name']}
                datas.append(dict_obj)
    df=pd.DataFrame(datas, columns=('xmin', 'ymin', 'xmax', 'ymax', 'label'))
    df.to_csv(prefix_dir+"train_data_dir.csv")
xml_to_csv_file()
