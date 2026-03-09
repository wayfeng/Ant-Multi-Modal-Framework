from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os

from nn4k.consts import NN_EXECUTOR_KEY
from nn4k.invoker import LLMInvoker

from PIL import Image
from scipy.special import softmax
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import openvino as ov

import argparse



class ImageEncoder(nn.Module):
    def __init__(self, model):
        super(ImageEncoder, self).__init__()
        self.backbone = model.backbone.eval()
        self.backbone_vl = model.backbone_vl.eval()
        self.itc_vl_image_proj = model.itc_vl_image_proj.eval()

    def forward(self, img):
        vffn_hiddens = self.backbone(visual_tokens=img)["encoder_out"]
        vlffn_hiddens = self.backbone_vl(
            src_tokens=None,
            token_embeddings=vffn_hiddens,
            multiway_split_position=-1,
        )["encoder_out"]

        cls_vlffn_feats = self.itc_vl_image_proj(vlffn_hiddens[:, 0])
        cls_vlffn_feats = cls_vlffn_feats / cls_vlffn_feats.norm(dim=-1, keepdim=True)
        return cls_vlffn_feats

class TextEncoder(nn.Module):
    def __init__(self, model):
        super(TextEncoder, self).__init__()
        self.backbone = model.backbone.eval()
        self.backbone_vl = model.backbone_vl.eval()
        self.itc_vl_text_proj = model.itc_vl_text_proj.eval()

    def forward(self, text_ids, text_masks):
        text_padding_position = 1 - text_masks
        lffn_hiddens = self.backbone(
            textual_tokens=text_ids,
            text_padding_position=text_padding_position,
        )["encoder_out"]
        vlffn_hiddens = self.backbone_vl(
            src_tokens=None,
            token_embeddings=lffn_hiddens,
            encoder_padding_mask=text_padding_position,
            multiway_split_position=-1,
        )["encoder_out"]

        cls_vlffn_feats = self.itc_vl_text_proj(vlffn_hiddens[:, 0])
        cls_vlffn_feats = cls_vlffn_feats / cls_vlffn_feats.norm(dim=-1, keepdim=True)

        return cls_vlffn_feats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PyTorch model to OpenVINO IR")
    parser.add_argument("--cfg", type=str, default="./configs/Encoder_0.4B.json", help="Path to model config file")
    args = parser.parse_args()

    model_name = args.cfg.split('/')[-1].replace('.json', '').replace('.', '_').lower()
    cfg = {
        'model_config': args.cfg,
        #'model_config': './configs/Encoder_1B.json',
        NN_EXECUTOR_KEY: 'm2_encoder.M2EncoderExecutor'
    }

    image_norm = transforms.Compose([transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    print(f"Loading PyTorch model {model_name} ...")
    encoder = LLMInvoker.from_config(cfg)
    encoder.warmup_local_model()

    model = encoder._nn_executor._model
    print("Logit scale:", model.logit_scale.exp().item())

    ie = ImageEncoder(model)
    te = TextEncoder(model)
    dummy_input = torch.randn((1, 3, 224, 224), dtype=torch.float32)

    image_path = './pics/pokemon.jpeg'
    img = encoder._nn_executor._img_processor(Image.open(image_path).convert('RGB')).unsqueeze(0)
    data_text = ["杰尼龟", "妙蛙种子", "小火龙", "皮卡丘", "卡车"]
    tokenizer = encoder._nn_executor._tokenizer
    max_length = model.hparams.config["max_text_len"]
    txt_encoding = tokenizer(
        data_text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_special_tokens_mask=True,
    )
    text_ids = torch.tensor(txt_encoding["input_ids"])
    text_masks = torch.tensor(txt_encoding["attention_mask"])

    img = image_norm(img)
    print("Testing PyTorch model...")
    with torch.no_grad():
        ifeats = ie(img)
        print(ifeats.shape)
        tfeats = te(text_ids, text_masks)
        print(tfeats.shape)
        logit_scale = model.logit_scale.exp()
        logits_per_img = logit_scale * ifeats @ tfeats.t()

        probs = logits_per_img.softmax(dim=-1).cpu().numpy()
        print(f"Original model output probs: {probs}")

    # To OV IR
    print("Converting to OpenVINO IR...")
    image_encoder_path = f"ov_models/{model_name}_image"
    text_encoder_path = f"ov_models/{model_name}_text"
    with torch.no_grad():
        if not os.path.exists("ov_models"):
            os.makedirs("ov_models")
        if os.path.exists(f"{image_encoder_path}.xml") and os.path.exists(f"{text_encoder_path}.xml"):
            print("OpenVINO IR already exists, skipping conversion.")
        else:
            ov_ie_model = ov.convert_model(ie, example_input=dummy_input, input=(-1, 3, 224, 224))
            ov.save_model(ov_ie_model, f"{image_encoder_path}.xml")
            ov_te_model = ov.convert_model(te, example_input=(text_ids, text_masks), input=[(-1, max_length), (-1, max_length)])
            ov.save_model(ov_te_model, f"{text_encoder_path}.xml")

    print("Verifying OpenVINO IR...")
    ov_ie_model = ov.Core().read_model(f"{image_encoder_path}.xml")
    ov_te_model = ov.Core().read_model(f"{text_encoder_path}.xml")
    ov_ie_compiled = ov.compile_model(ov_ie_model, device_name="GPU")
    ov_te_compiled = ov.compile_model(ov_te_model, device_name="GPU")

    ov_ifeats = ov_ie_compiled(img.cpu().numpy()).to_tuple()[0]
    ov_tfeats = ov_te_compiled([text_ids.cpu().numpy(), text_masks.cpu().numpy()]).to_tuple()[0]
    logit_scale = model.logit_scale.exp().cpu().detach().numpy()
    print("Logit scale:", logit_scale)
    ov_logits_per_img = logit_scale * ov_ifeats @ ov_tfeats.T
    ov_probs = softmax(ov_logits_per_img, axis=-1)
    print(f"OpenVINO IR output probs: {ov_probs}")
