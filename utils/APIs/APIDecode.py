'''
decode api: 将model预测出的数据转变成理想的格式
    tips: 
        需要与Trainer配合, Trainer predict的输出即此api输入
'''

from tqdm import tqdm


def api_decode(outputs, labelvocab):
    formated_outputs = ['guid,tag']
    for guid, label in tqdm(outputs, desc='----- [Decoding]'):
        formated_outputs.append((str(guid) + ',' + labelvocab.id_to_label(label)))
    return formated_outputs