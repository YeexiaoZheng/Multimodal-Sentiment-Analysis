'''
普通的常用工具

'''

import os
import json
import chardet
import torch
from tqdm import tqdm
from PIL import Image
from sklearn.model_selection import train_test_split

# 将文本和标签格式化成一个json
def data_format(input_path, data_dir, output_path):
    data = []
    with open(input_path) as f:
        for line in tqdm(f.readlines(), desc='----- [Formating]'):
            guid, label = line.replace('\n', '').split(',')
            text_path = os.path.join(data_dir, (guid + '.txt'))
            if guid == 'guid': continue
            with open(text_path, 'rb') as textf:
                text_byte = textf.read()
                encode = chardet.detect(text_byte)
                try:
                    text = text_byte.decode(encode['encoding'])
                except:
                    # print('can\'t decode file', guid)
                    try:
                        text = text_byte.decode('iso-8859-1').encode('iso-8859-1').decode('gbk')
                    except:
                        print('not is0-8859-1', guid)
                        continue
            text = text.strip('\n').strip('\r').strip(' ').strip()
            data.append({
                'guid': guid,
                'label': label,
                'text': text
            })
    with open(output_path, 'w') as wf:
        json.dump(data, wf, indent=4)

# 读取数据，返回[(guid, text, img, label)]元组列表
def read_from_file(path, data_dir, only=None):
    data = []
    with open(path) as f:
        json_file = json.load(f)
        for d in tqdm(json_file, desc='----- [Loading]'):
            guid, label, text = d['guid'], d['label'], d['text']
            if guid == 'guid': continue

            if only == 'text': img = Image.new(mode='RGB', size=(224, 224), color=(0, 0, 0))
            else:
                img_path = os.path.join(data_dir, (guid + '.jpg'))
                # img = cv2.imread(img_path)
                img = Image.open(img_path)
                img.load()

            if only == 'img': text = ''

            data.append((guid, text, img, label))
        f.close()

    return data


# 分离训练集和验证集
def train_val_split(data, val_size=0.2):
    return train_test_split(data, train_size=(1-val_size), test_size=val_size)


# 写入数据
def write_to_file(path, outputs):
    with open(path, 'w') as f:
        for line in tqdm(outputs, desc='----- [Writing]'):
            f.write(line)
            f.write('\n')
        f.close()


# 保存模型
def save_model(output_path, model_type, model):
    output_model_dir = os.path.join(output_path, model_type)
    if not os.path.exists(output_model_dir): os.makedirs(output_model_dir)    # 没有文件夹则创建
    model_to_save = model.module if hasattr(model, 'module') else model     # Only save the model it-self
    output_model_file = os.path.join(output_model_dir, "pytorch_model.bin")
    torch.save(model_to_save.state_dict(), output_model_file)