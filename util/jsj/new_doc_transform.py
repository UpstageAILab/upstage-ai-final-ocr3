import os
import json
from tqdm import tqdm

def transform_label(data_path, json_path, outpath=None):
    new_dic={'images':{}}
    file_names = []
    for cat1 in os.listdir(f"{data_path}/"):
        jpg_path = f"{data_path}/{cat1}"
        for cat2 in os.listdir(jpg_path):
            jpg_path = f"{data_path}/{cat1}/{cat2}"
            for year in os.listdir(jpg_path):
                jpg_path = f"{data_path}/{cat1}/{cat2}/{year}"
                for file_name in os.listdir(jpg_path):
                    file_name_com = file_name.replace(".jpg","").split("-")
                    file_names.append((cat1,file_name_com))

    for cat1,file_name_com in tqdm(file_names):
        with open(f"{json_path}/{cat1}/{file_name_com[0]}/{file_name_com[1]}/{'-'.join(file_name_com)}.json")as f:
            data_dic=json.load(f)
        file_name = f"{'-'.join(file_name_com)}.jpg"
        new_dic['images'][file_name] = {'words':{}}
        for annotation in data_dic['annotations']:
            new_words = {}
            bbox = annotation['annotation.bbox']
            new_words['points'] = [
                [bbox[0],bbox[1]],
                [bbox[0] + bbox[2], bbox[1]],
                [bbox[0] + bbox[2], bbox[1] + bbox[3]],
                [bbox[0],bbox[1] + bbox[3]],
            ]
            
            if annotation['annotation.ttype'] == 'textType1':
                new_words['orientation'] = 'Horizontal'
            else:
                assert Exception("New ttype")
            new_words['language'] = ['ko']
            new_dic['images'][file_name]['words'][f"{annotation['id']+1}".zfill(4)]=new_words
    outpath = outpath if outpath is not None else "./"
    with open(f"{outpath}/new.json", "w") as f:
        json.dump(new_dic,f,ensure_ascii=False)
if __name__=="__main__":
    transform_label(
        "/data/datasets/032.공공행정문서_OCR/01.데이터/01.Training/02.원천데이터(jpg)",
        "/data/datasets/032.공공행정문서_OCR/01.데이터/01.Training/01.라벨링데이터(Json)",
    )