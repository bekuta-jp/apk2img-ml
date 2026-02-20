from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tv


def _get_first_conv_and_setter(model: nn.Module):
    if hasattr(model, "conv1") and isinstance(model.conv1, nn.Conv2d):
        return model.conv1, (lambda new: setattr(model, "conv1", new))

    if hasattr(model, "features"):
        first = model.features[0]
        if isinstance(first, nn.Conv2d):
            return first, (lambda new: model.features.__setitem__(0, new))
        if hasattr(first, "__getitem__") and len(first) > 0 and isinstance(first[0], nn.Conv2d):
            return first[0], (lambda new: first.__setitem__(0, new))

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            parent_name = ".".join(name.split(".")[:-1])
            attr = name.split(".")[-1]
            parent = model.get_submodule(parent_name) if parent_name else model

            def setter(new_layer, *, parent=parent, attr=attr):
                if attr.isdigit():
                    parent[int(attr)] = new_layer
                else:
                    setattr(parent, attr, new_layer)

            return module, setter

    raise RuntimeError("no Conv2d layer found")


def _adapt_first_conv(model: nn.Module, in_channels: int) -> nn.Module:
    conv, setter = _get_first_conv_and_setter(model)
    new_conv = nn.Conv2d(
        in_channels,
        conv.out_channels,
        conv.kernel_size,
        conv.stride,
        conv.padding,
        conv.dilation,
        conv.groups,
        bias=(conv.bias is not None),
        padding_mode=conv.padding_mode,
    )

    with torch.no_grad():
        weight = conv.weight
        orig_in = weight.shape[1]

        if in_channels == orig_in:
            new_conv.weight.copy_(weight)
        elif in_channels < orig_in:
            if in_channels == 1:
                new_conv.weight.copy_(weight.mean(dim=1, keepdim=True))
            else:
                new_conv.weight[:, :in_channels].copy_(weight[:, :in_channels])
                new_conv.weight[:, in_channels:].zero_()
        else:
            rep = in_channels // orig_in
            rem = in_channels % orig_in
            expanded = weight.repeat(1, rep, 1, 1)
            if rem > 0:
                expanded = torch.cat([expanded, weight[:, :rem]], dim=1)
            new_conv.weight.copy_(expanded[:, :in_channels])

        if conv.bias is not None:
            new_conv.bias.copy_(conv.bias)

    setter(new_conv)
    return model


def _replace_classifier(model: nn.Module, out_features: int) -> nn.Module:
    # DenseNet exposes a plain Linear classifier.
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear):
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, out_features)
        return model

    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
        last = model.classifier[-1]
        if isinstance(last, nn.Linear):
            model.classifier[-1] = nn.Linear(last.in_features, out_features)
            return model

    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        model.fc = nn.Linear(model.fc.in_features, out_features)
        return model

    raise RuntimeError("unsupported classifier layout")


def get_model(name: str, *, num_classes: int = 2, pretrained: bool = True, in_channels: int = 1) -> nn.Module:
    key = name.lower()

    if key == "tiny":
        return Tiny3Conv(num_classes=num_classes, in_channels=in_channels)

    if key == "alexnet":
        model = tv.alexnet(weights=tv.AlexNet_Weights.IMAGENET1K_V1 if pretrained else None)
    elif key == "vgg16":
        model = tv.vgg16_bn(weights=tv.VGG16_BN_Weights.IMAGENET1K_V1 if pretrained else None)
    elif key == "resnet50":
        model = tv.resnet50(weights=tv.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
    elif key == "densenet":
        model = tv.densenet121(weights=tv.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
    elif key == "mobilenet":
        model = tv.mobilenet_v2(weights=tv.MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None)
    else:
        raise ValueError(f"unknown model name: {name}")

    model = _adapt_first_conv(model, in_channels)
    model = _replace_classifier(model, num_classes)
    return model


class Tiny3Conv(nn.Module):
    def __init__(self, *, num_classes: int = 2, in_channels: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 32 * 32, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):  # noqa: D401
        return self.classifier(self.features(x))
