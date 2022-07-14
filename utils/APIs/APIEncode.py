'''
encode api: 将原始data数据转化成APIDataset所需要的数据
    tips:
        ! 必须调用labelvocab的add_label接口将标签加入labelvocab字典
'''

from tqdm import tqdm
from transformers import AutoTokenizer
from torchvision import transforms


def api_encode(data, labelvocab, config):

    ''' 这里直接加入三个标签, 后面就不需要添加了 '''
    labelvocab.add_label('positive')
    labelvocab.add_label('neutral')
    labelvocab.add_label('negative')
    labelvocab.add_label('null')    # 空标签

    ''' 文本处理 BERT的tokenizer '''
    tokenizer = AutoTokenizer.from_pretrained(config.bert_name)

    ''' 图像处理 torchvision的transforms '''
    def get_resize(image_size):
        for i in range(20):
            if 2**i >= image_size:
                return 2**i
        return image_size

    img_transform = transforms.Compose([
                transforms.Resize(get_resize(config.image_size)),
                transforms.CenterCrop(config.image_size),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    ''' 对读入的data进行预处理 '''
    guids, encoded_texts, encoded_imgs, encoded_labels = [], [], [], []
    for line in tqdm(data, desc='----- [Encoding]'):
        guid, text, img, label = line
        # id
        guids.append(guid)
        
        # 文本
        text.replace('#', '')
        tokens = tokenizer.tokenize('[CLS]' + text + '[SEP]')
        encoded_texts.append(tokenizer.convert_tokens_to_ids(tokens))

        # 图像
        encoded_imgs.append(img_transform(img))
        
        # 标签
        encoded_labels.append(labelvocab.label_to_id(label))

    return guids, encoded_texts, encoded_imgs, encoded_labels

