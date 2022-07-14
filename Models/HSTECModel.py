import torch
import torch.nn as nn
from transformers import AutoModel
from torchvision.models import resnet50

class TextModel(nn.Module):

    def __init__(self, config):
        super(TextModel, self).__init__()

        self.bert = AutoModel.from_pretrained(config.bert_name)
        self.trans = nn.Sequential(
            nn.Dropout(config.bert_dropout),
            nn.Linear(self.bert.config.hidden_size, config.middle_hidden_size),
            nn.ReLU(inplace=True)
        ) 
        
        # 是否进行fine-tune
        for param in self.bert.parameters():
            if config.fixed_text_model_params:
                param.requires_grad = False
            else:
                param.requires_grad = True
        
        # self.bert.init_weights()

    def forward(self, bert_inputs, masks, token_type_ids=None):
        bert_out = self.bert(input_ids=bert_inputs, token_type_ids=token_type_ids, attention_mask=masks)
        hidden_state = bert_out['last_hidden_state']
        pooler_out = bert_out['pooler_output']

        # print(hidden_state.shape, pooler_out.shape)
        
        return self.trans(hidden_state), self.trans(pooler_out)


class ImageModel(nn.Module):

    def __init__(self, config):
        super(ImageModel, self).__init__()
        self.full_resnet = resnet50(pretrained=True)
        self.resnet_h = nn.Sequential(
            *(list(self.full_resnet.children())[:-2]),
        )

        self.resnet_p = nn.Sequential(
            list(self.full_resnet.children())[-2],
            nn.Flatten()
        )

        # 本层为特殊层，目的是为了得到较少的特征响应图(原来的2048有些过大)：
        # (batch, 2048, 7, 7) -> (batch, img_hidden_seq, middle_hidden_size)
        self.hidden_trans = nn.Sequential(
            nn.Conv2d(self.full_resnet.fc.in_features, config.img_hidden_seq, 1),
            nn.Flatten(start_dim=2),
            nn.Dropout(config.resnet_dropout),
            nn.Linear(7 * 7, config.middle_hidden_size),    # 这里的7*7是根据resnet50，原img大小为224*224的情况来的
            nn.ReLU(inplace=True)
        )

        self.trans = nn.Sequential(
            nn.Dropout(config.resnet_dropout),
            nn.Linear(self.full_resnet.fc.in_features, config.middle_hidden_size),
            nn.ReLU(inplace=True)
        )
        
        # 是否进行fine-tune
        for param in self.full_resnet.parameters():
            if config.fixed_image_model_params:
                param.requires_grad = False
            else:
                param.requires_grad = True

    def forward(self, imgs):
        hidden_state = self.resnet_h(imgs)
        feature = self.resnet_p(hidden_state)

        # print(hidden_state.shape, feature.shape)

        return self.hidden_trans(hidden_state), self.trans(feature)


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        # text
        self.text_model = TextModel(config)
        # image
        self.img_model = ImageModel(config)
        # attention
        self.attention = nn.TransformerEncoderLayer(
            d_model=config.middle_hidden_size,
            nhead=config.attention_nhead, 
            dropout=config.attention_dropout
        )
        
        # 全连接分类器
        self.text_classifier = nn.Sequential(
            nn.Dropout(config.fuse_dropout),
            nn.Linear(config.middle_hidden_size * 2, config.out_hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(config.fuse_dropout),
            nn.Linear(config.out_hidden_size, config.num_labels),
            nn.Softmax(dim=1)
        )
        self.img_classifier = nn.Sequential(
            nn.Dropout(config.fuse_dropout),
            nn.Linear(config.middle_hidden_size * 2, config.out_hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(config.fuse_dropout),
            nn.Linear(config.out_hidden_size, config.num_labels),
            nn.Softmax(dim=1)
        )
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, texts, texts_mask, imgs, labels=None):
        text_hidden_state, text_feature = self.text_model(texts, texts_mask)

        img_hidden_state, img_feature = self.img_model(imgs)

        text_hidden_state = text_hidden_state.permute(1, 0, 2)
        img_hidden_state = img_hidden_state.permute(1, 0, 2)

        attention_out = self.attention(torch.cat([text_hidden_state, img_hidden_state], dim=0))
        
        attention_out = torch.mean(attention_out, dim=0).squeeze(0)

        text_prob_vec = self.text_classifier(torch.cat([text_feature, attention_out], dim=1))
        img_prob_vec = self.img_classifier(torch.cat([img_feature, attention_out], dim=1))

        prob_vec = torch.softmax((text_prob_vec + img_prob_vec), dim=1)
        pred_labels = torch.argmax(prob_vec, dim=1)

        if labels is not None:
            loss = self.loss_func(prob_vec, labels)
            return pred_labels, loss
        else:
            return pred_labels