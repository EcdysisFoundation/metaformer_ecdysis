import argparse
import csv
import logging
from pathlib import Path
from typing import Union

import torch
from PIL import Image
from torch.nn import Softmax
from torchvision.transforms import transforms, InterpolationMode
from yacs.config import CfgNode

from build import build_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetaformerInferencer:
    """Inference class for metaformer model"""

    def __init__(self, device: str = None):
        self.device = torch.device(device) if device is not None else torch.device('cpu')
        self._config = None
        self._output_function = None
        self.class_names = None
        self.transform = None
        self.model = None

    @property
    def config(self) -> CfgNode:
        return self._config

    @config.setter
    def config(self, value):
        if isinstance(value, CfgNode):
            self._config = value
        elif isinstance(value, Path):
            self._config = CfgNode.load_cfg(open(value, "r"))
        else:
            raise ValueError('Type of config must be either yacs.config.CfgNode or pathlib.Path')


    @property
    def output_function(self) -> callable:
        return self._output_function

    @output_function.setter
    def output_function(self, value):
        if value is None:
            self._output_function = None
            return

        value = value.lower()
        if value == 'softmax':
            self._output_function = Softmax(dim=1)
        else:
            raise ValueError('output_func must be either None or "softmax"')

    def build(self, config: Union[Path, CfgNode], checkpoint: Path, output_function: str = None):
        """
        Build pipeline to perform inference
        """
        self.config = config
        self.class_names = self.config.DATA.CLASS_NAMES
        self.model = build_model(self.config).to(self.device)
        checkpoint = torch.load(checkpoint, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'], strict=False)
        self.model.eval()
        self.transform = self.make_transform()
        self.output_function = output_function

    def make_transform(self):
        """
        Create image transformation as in training. The transformation is a composition of Resizing to model's input
        image size -> Conversion to torch tensor -> Normalization using Imagenet's mean and std.

        Returns: Transformation callable
        """

        imagenet_default_mean = (0.485, 0.456, 0.406)
        imagenet_default_std = (0.229, 0.224, 0.225)

        image_size = self.config.DATA.IMG_SIZE

        transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_default_mean, imagenet_default_std)
        ])

        return transform

    @torch.no_grad()
    def __call__(self, image: Union[Image.Image, torch.tensor]) -> torch.Tensor:
        if self.model is None:
            raise ValueError('Model is not built. Call build() method first')

        image = self.transform(image)
        if image.dim() == 3:
            image = image.unsqueeze(0)
        image = image.to(self.device)

        predictions = self.model(image)

        if self.output_function is not None:
            predictions = self.output_function(predictions)

        return predictions


def _get_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description='Metaformer inference script')

    parser.add_argument('model_name', type=str, help='Name of the model to use')
    parser.add_argument('image_path', type=str, help='Path to the input image')

    parser.add_argument('--model-type', type=str, default='MetaFG_2', help='Type of the model, i.e. MetaFG_2')

    return parser.parse_args()


def load_mapping(map_file: Path):
    with open(map_file) as f:
        next(f)  # Skip header, note expected order
        mapping = {morphospecies_id: morphospecies_name for morphospecies_id, morphospecies_name in csv.reader(f)}
        return mapping


def _main():
    args = _get_args()

    model_root = Path('output')/args.model_type/args.model_name

    config = model_root/'config.yaml'
    checkpoint = model_root/'best.pth'

    inferencer = MetaformerInferencer(device='cuda:0')
    inferencer.build(config, checkpoint, output_function='softmax')
    image = Image.open(args.image_path)

    mapping = load_mapping(Path('deploy/morphospecies_map.csv'))
    predictions = inferencer(image)

    prediction_index = predictions.argmax().item()
    prediction_class = inferencer.class_names[prediction_index]
    prediction_taxon = mapping.get(prediction_class, 0)

    logger.info(f'\n\tPrediction:\n\tIndex: {prediction_index} | Class: {prediction_class} |'
                f' Taxon: {prediction_taxon} | Probability: {predictions.max().item():.4f}')


if __name__ == '__main__':
    _main()
