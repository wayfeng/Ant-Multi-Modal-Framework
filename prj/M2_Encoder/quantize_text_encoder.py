import argparse
from zipfile import ZipFile
import openvino as ov
import logging
import nncf
import json
import torch
import torch.utils.data as data
from pathlib import Path

from nn4k.consts import NN_EXECUTOR_KEY
from nn4k.invoker import LLMInvoker
from quantize import load_data

def transform_fn(text_data):
    """
    Quantization transform function. Extracts and preprocess input data from dataloader item for quantization.
    Parameters:
        text_data: text data produced by DataLoader during iteration
    Returns:
        input_tensor: input data in Dict format for model quantization
    """
    input_tensor = {
        "text_ids": text_data["input_ids"].squeeze(0),
        "text_masks": text_data["attention_mask"].squeeze(0)
    }
    return input_tensor

class COCOLabelsLoader(data.Dataset):
    def __init__(self, labels_path, tokenizer, max_length):
        labels_path = Path(labels_path)
        if not labels_path.is_file():
            raise ValueError(f"labels_path must be a JSON or JSONL file, got: {labels_path}")
        self.labels = self._load_labels(labels_path, tokenizer, max_length)

    @staticmethod
    def _to_label(item):
        if isinstance(item, str):
            return item.strip()
        if isinstance(item, dict):
            for key in ("caption", "text", "label"):
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
                elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], str) and value[0].strip():
                    return value[0].strip()
        return None

    def _load_labels(self, labels_path: Path, tokenizer, max_length):
        suffix = labels_path.suffix.lower()
        assert suffix == ".jsonl", f"Only .jsonl format is supported for labels, got: {suffix}"

        records = []
        with open(labels_path, 'r', encoding='utf-8') as f:
            records = [json.loads(line) for line in f if line.strip()]

        labels = []
        for item in records:
            label = self._to_label(item)
            if label:
                text_data = tokenizer(label, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
                labels.append(text_data)
        return labels

    def __getitem__(self, index):
        return self.labels[index]

    def __len__(self):
        return len(self.labels)


def load_data(data_dir: Path, tokenizer, max_length):
    if not (data_dir / "coco-cn_test.jsonl").exists():
        import urllib.request
        DATA_URL = "https://ultralytics.com/assets/coco128.zip"
        zipfile = data_dir/'coco128.zip'
        print(f"Downloading {DATA_URL} to {zipfile}...")
        urllib.request.urlretrieve(DATA_URL, zipfile)
        with ZipFile(zipfile, "r") as zip_ref:
            zip_ref.extractall(data_dir)

    coco_dataset = COCOLabelsLoader(data_dir / 'coco-cn_test.jsonl', tokenizer, max_length)
    calibration_loader = torch.utils.data.DataLoader(coco_dataset)

    return nncf.Dataset(calibration_loader, transform_fn)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='ov_models/encoder_0_4b_text.xml',
                        help="fp16 model to quantize")
    parser.add_argument('-d', '--data', default='./data/',
                        help="Data folder to calibrate model")
    parser.add_argument("--cfg", type=str, default="./configs/Encoder_0.4B.json", help="Path to model config file")
    args = parser.parse_args()

    model_name = args.cfg.split('/')[-1].replace('.json', '').replace('.', '_').lower()
    cfg = {
        'model_config': args.cfg,
        NN_EXECUTOR_KEY: 'm2_encoder.M2EncoderExecutor'
    }
    encoder = LLMInvoker.from_config(cfg)
    encoder.warmup_local_model()
    max_length = encoder._nn_executor._model.hparams.config["max_text_len"]
    tokenizer = encoder._nn_executor._tokenizer

    core = ov.Core()

    nncf.set_log_level(logging.ERROR)
    fp16_model_path = Path(args.model)
    int8_model_path = Path(args.model).with_name(f"{Path(args.model).stem}_int8.xml")
    # calibration_data = prepare_dataset()
    ov_model = core.read_model(fp16_model_path)
    calibration_dataset = load_data(Path(args.data), tokenizer, max_length)
    quantized_model = nncf.quantize(
        model=ov_model,
        calibration_dataset=calibration_dataset,
        model_type=nncf.ModelType.TRANSFORMER,
    )
    ov.save_model(quantized_model, int8_model_path)