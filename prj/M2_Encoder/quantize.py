import argparse
import openvino as ov
import logging
import cv2
import nncf
import numpy as np
import torch
import torch.utils.data as data
from torchvision.transforms.functional import to_pil_image
from zipfile import ZipFile
from pathlib import Path
from PIL import Image

def normalize(arr, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]):
    arr = arr.astype(np.float32)
    arr /= 255.0
    for i in range(3):
        arr[...,i] = (arr[...,i] - mean[i]) / std[i]
    return arr

def preprocess_image(img, shape=[224,224]):
    img = cv2.resize(img, tuple(shape), interpolation=cv2.INTER_NEAREST)
    img = normalize(np.asarray(img))
    return img.transpose(2,0,1)

def transform_fn(image_data):
    """
    Quantization transform function. Extracts and preprocess input data from dataloader item for quantization.
    Parameters:
        image_data: image data produced by DataLoader during iteration
    Returns:
        input_tensor: input data in Dict format for model quantization
    """
    img = preprocess_image(image_data.numpy().squeeze())
    return torch.from_numpy(img).unsqueeze(0)


class COCOLoader(data.Dataset):
    def __init__(self, images_path):
        self.images = list(Path(images_path).iterdir())

    def __getitem__(self, index):
        image_path = self.images[index]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def __len__(self):
        return len(self.images)


def load_data(data_dir: Path):
    if not (data_dir / "coco128/images/train2017").exists():
        import urllib.request
        DATA_URL = "https://ultralytics.com/assets/coco128.zip"
        zipfile = data_dir/'coco128.zip'
        print(f"Downloading {DATA_URL} to {zipfile}...")
        urllib.request.urlretrieve(DATA_URL, zipfile)
        with ZipFile(zipfile, "r") as zip_ref:
            zip_ref.extractall(data_dir)

    coco_dataset = COCOLoader(data_dir / 'coco128/images/train2017')
    calibration_loader = torch.utils.data.DataLoader(coco_dataset)

    return nncf.Dataset(calibration_loader, transform_fn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='ov_models/encoder_0_4b_image.xml',
                        help="fp16 model to quantize")
    parser.add_argument('-d', '--data', default='./data',
                        help="Data folder to calibrate model")
    args = parser.parse_args()

    core = ov.Core()

    nncf.set_log_level(logging.ERROR)
    fp16_model_path = Path(args.model)
    int8_model_path = Path(args.model).with_name(f"{Path(args.model).stem}_int8.xml")
    # calibration_data = prepare_dataset()
    ov_model = core.read_model(fp16_model_path)
    calibration_dataset = load_data(Path(args.data))
    quantized_model = nncf.quantize(
        model=ov_model,
        calibration_dataset=calibration_dataset,
        model_type=nncf.ModelType.TRANSFORMER,
    )
    ov.save_model(quantized_model, int8_model_path)
