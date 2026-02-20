# models_zoo.py
import torch
import torch.nn as nn, torchvision.models as tv

def _get_first_conv_and_setter(model):
    """最初のConv2dモジュールと、それを置き換えるsetterを返す（各モデル構造に耐性あり）"""
    # ResNet系
    if hasattr(model, "conv1") and isinstance(model.conv1, nn.Conv2d):
        return model.conv1, (lambda new: setattr(model, "conv1", new))
    # VGG/AlexNet系（featuresの先頭がConv2d）
    if hasattr(model, "features"):
        if isinstance(model.features[0], nn.Conv2d):
            return model.features[0], (lambda new: model.features.__setitem__(0, new))
        # MobileNetV2など：features[0] がブロック、その中の最初のConv2d
        if hasattr(model.features[0], "0") and isinstance(model.features[0][0], nn.Conv2d):
            def setter(new):
                block = model.features[0]
                block[0] = new
            return model.features[0][0], setter
    # フォールバック：最初に見つかったConv2dを置換（念のため）
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            parent_name = ".".join(name.split(".")[:-1])
            last = name.split(".")[-1]
            parent = model.get_submodule(parent_name) if parent_name else model
            def setter(new, parent=parent, last=last):
                if last.isdigit():
                    parent[int(last)] = new
                else:
                    setattr(parent, last, new)
            return m, setter
    raise RuntimeError("No Conv2d layer found as first conv.")

def _adapt_first_conv(model, in_ch:int):
    """最初のConvを任意チャンネル入力に対応させる。
       - in_ch == orig: 重みコピー
       - in_ch  < orig: in_ch==1なら平均化、それ以外は先頭in_chチャネルを使用
       - in_ch  > orig: 既存重みを繰り返して埋める
    """
    conv, setter = _get_first_conv_and_setter(model)
    new = nn.Conv2d(in_ch, conv.out_channels, conv.kernel_size,
                    conv.stride, conv.padding, conv.dilation,
                    conv.groups, bias=(conv.bias is not None), padding_mode=conv.padding_mode)
    with torch.no_grad():
        W = conv.weight  # (out_c, in_c, kH, kW)
        orig = W.shape[1]
        if in_ch == orig:
            new.weight.copy_(W)
        elif in_ch < orig:
            if in_ch == 1:
                new.weight.copy_(W.mean(dim=1, keepdim=True))
            else:
                new.weight[:, :in_ch].copy_(W[:, :in_ch])
                # 余りは0初期化（明示的に安定）
                if in_ch < new.weight.shape[1]:
                    new.weight[:, in_ch:].zero_()
        else:  # in_ch > orig
            # 繰り返し＋端数で埋める
            rep = in_ch // orig
            rem = in_ch % orig
            w = W.repeat(1, rep, 1, 1)
            if rem > 0:
                w = torch.cat([w, W[:, :rem]], dim=1)
            new.weight.copy_(w[:, :in_ch])
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    setter(new)
    return model

def _replace_fc(model, out_features):
    """最後の FC を (in→out_features) に置換"""
    if hasattr(model,'classifier') and isinstance(model.classifier, nn.Sequential):
        in_f = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_f, out_features)
    elif hasattr(model,'fc'):
        in_f = model.fc.in_features
        model.fc = nn.Linear(in_f, out_features)
    return model

def get_model(name:str, num_classes:int=2, pretrained:bool=True, in_ch:int=1):
    name = name.lower()
    if name=='tiny':
        return Tiny3Conv(num_classes, in_ch=in_ch)
    if name=='alexnet':
        m = tv.alexnet(weights=(tv.AlexNet_Weights.IMAGENET1K_V1 if pretrained else None))
    elif name=='vgg16':
        m = tv.vgg16_bn(weights=(tv.VGG16_BN_Weights.IMAGENET1K_V1 if pretrained else None))
    elif name=='resnet50':
        m = tv.resnet50(weights=(tv.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None))
    elif name=='densenet':
        m = tv.densenet121(weights=(tv.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None))
    elif name=='mobilenet':
        m = tv.mobilenet_v2(weights=(tv.MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None))
    else:
        raise ValueError(f"unknown model {name}")
    m = _adapt_first_conv(m, in_ch)
    m = _replace_fc(m, num_classes)
    return m

class Tiny3Conv(nn.Module):
    """論文 baseline 相当 (Conv→ReLU→Pool×3 + FC) / 任意in_ch対応"""
    def __init__(self, num_cls=2, in_ch=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),   nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1),  nn.ReLU(), nn.MaxPool2d(2))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*32*32,256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256,num_cls))
    def forward(self,x): return self.classifier(self.features(x))
