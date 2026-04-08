from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tv

SUPPORTED_MODEL_NAMES = (
    "tiny",
    "alexnet",
    "vgg16",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "densenet",
    "densenet121",
    "mobilenet",
    "mobilenet_v2",
    "efficientnet_b0",
    "efficientnet_b1",
    "efficientnet_b2",
    "efficientnet_b3",
    "efficientnet_b4",
    "efficientnet_b5",
    "efficientnet_b6",
    "efficientnet_b7",
    "efficientnet_v2_s",
    "efficientnet_v2_m",
    "efficientnet_v2_l",
)


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


def _normalize_model_name(name: str) -> str:
    key = name.lower().replace("-", "_")
    aliases = {
        "vgg16_bn": "vgg16",
        "efficientnet_v2s": "efficientnet_v2_s",
        "efficientnet_v2m": "efficientnet_v2_m",
        "efficientnet_v2l": "efficientnet_v2_l",
        **{f"efficientnetb{idx}": f"efficientnet_b{idx}" for idx in range(8)},
    }
    return aliases.get(key, key)


def _make_model_from_torchvision(key: str, pretrained: bool) -> nn.Module:
    builders = {
        "alexnet": (tv.alexnet, tv.AlexNet_Weights.DEFAULT),
        "vgg16": (tv.vgg16_bn, tv.VGG16_BN_Weights.DEFAULT),
        "resnet18": (tv.resnet18, tv.ResNet18_Weights.DEFAULT),
        "resnet34": (tv.resnet34, tv.ResNet34_Weights.DEFAULT),
        "resnet50": (tv.resnet50, tv.ResNet50_Weights.DEFAULT),
        "resnet101": (tv.resnet101, tv.ResNet101_Weights.DEFAULT),
        "resnet152": (tv.resnet152, tv.ResNet152_Weights.DEFAULT),
        "densenet": (tv.densenet121, tv.DenseNet121_Weights.DEFAULT),
        "densenet121": (tv.densenet121, tv.DenseNet121_Weights.DEFAULT),
        "mobilenet": (tv.mobilenet_v2, tv.MobileNet_V2_Weights.DEFAULT),
        "mobilenet_v2": (tv.mobilenet_v2, tv.MobileNet_V2_Weights.DEFAULT),
        "efficientnet_b0": (tv.efficientnet_b0, tv.EfficientNet_B0_Weights.DEFAULT),
        "efficientnet_b1": (tv.efficientnet_b1, tv.EfficientNet_B1_Weights.DEFAULT),
        "efficientnet_b2": (tv.efficientnet_b2, tv.EfficientNet_B2_Weights.DEFAULT),
        "efficientnet_b3": (tv.efficientnet_b3, tv.EfficientNet_B3_Weights.DEFAULT),
        "efficientnet_b4": (tv.efficientnet_b4, tv.EfficientNet_B4_Weights.DEFAULT),
        "efficientnet_b5": (tv.efficientnet_b5, tv.EfficientNet_B5_Weights.DEFAULT),
        "efficientnet_b6": (tv.efficientnet_b6, tv.EfficientNet_B6_Weights.DEFAULT),
        "efficientnet_b7": (tv.efficientnet_b7, tv.EfficientNet_B7_Weights.DEFAULT),
        "efficientnet_v2_s": (tv.efficientnet_v2_s, tv.EfficientNet_V2_S_Weights.DEFAULT),
        "efficientnet_v2_m": (tv.efficientnet_v2_m, tv.EfficientNet_V2_M_Weights.DEFAULT),
        "efficientnet_v2_l": (tv.efficientnet_v2_l, tv.EfficientNet_V2_L_Weights.DEFAULT),
    }

    try:
        builder, weights = builders[key]
    except KeyError as exc:
        supported = ", ".join(SUPPORTED_MODEL_NAMES)
        raise ValueError(f"unknown model name: {key}. supported: {supported}") from exc

    return builder(weights=weights if pretrained else None)


def get_model(
    name: str,
    *,
    num_classes: int = 2,
    pretrained: bool = True,
    in_channels: int = 1,
    in_ch: int | None = None,
) -> nn.Module:
    if in_ch is not None:
        in_channels = in_ch

    key = _normalize_model_name(name)

    if key == "tiny":
        return Tiny3Conv(num_classes=num_classes, in_channels=in_channels)

    model = _make_model_from_torchvision(key, pretrained)
    model = _adapt_first_conv(model, in_channels)
    model = _replace_classifier(model, num_classes)
    return model


class Tiny3Conv(nn.Module):
    def __init__(
        self,
        *,
        num_classes: int = 2,
        in_channels: int = 1,
        in_ch: int | None = None,
    ):
        super().__init__()
        if in_ch is not None:
            in_channels = in_ch
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
        # Preserve the legacy 256x256 path while allowing arbitrary input sizes.
        self.pool = nn.AdaptiveAvgPool2d((32, 32))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 32 * 32, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):  # noqa: D401
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)
