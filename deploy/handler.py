import base64
import csv
import io
import logging
from argparse import Namespace
from pathlib import Path

import torch
from PIL import Image
from torch.nn import Softmax
from torchvision.transforms import transforms
from torchvision.transforms.functional import InterpolationMode

from ts.torch_handler.vision_handler import VisionHandler

from config import get_inference_config
from build import build_model

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


class MetaformerHandler(VisionHandler):
    """Torchserve handler for the MetaFormer model """

    def __init__(self):
        super(MetaformerHandler, self).__init__()
        self.config = None
        self.model_pt_path = None
        self.device = 'cpu'
        self.transform_image = None
        self.softmax = Softmax(dim=1)
        self.taxons = [*csv.DictReader(open('taxon_map.csv'))]

    def get_config(self, filename: str = 'config'):
        """
        Generate configuration object for the Metaformer model
        Args:
            filename: Name of the configuration yaml file
        """
        args = Namespace(cfg=f'{filename}.yaml')
        self.config = get_inference_config(args)

    def create_model(self):
        """
        Build model and load weights
        """
        self.model = build_model(self.config)
        checkpoint = torch.load(self.model_pt_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'], strict=False)
        self.model.eval()

    def get_context(self, context):
        """
        Process context object
        Args:
            context: Context contains model server system properties.
        """
        self.context = context
        self.manifest = context.manifest

        properties = context.system_properties
        model_dir = Path(properties.get("model_dir"))
        serialized_file = self.manifest['model']['serializedFile']
        self.model_pt_path = model_dir / serialized_file

    def initialize(self, context):
        """
        Initialize model. This method is only called when a new worker is added to this model in TorchServe
        Args:
            context: Initial context contains model server system properties, this is Torchserve specific.
        """

        self.get_context(context)
        self.get_config()

        self.create_model()

        self.transform_image = self.create_transform()

        self.initialized = True

    def create_transform(self):
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

    def preprocess(self, data):
        """The preprocess function converts the input data to a float tensor
        Args:
            data (List): Input data from the request is in the form of a Tensor
        Returns:
            list : The preprocess function returns the input image as a list of float tensors.
        """
        images = []

        for row in data:
            # Compat layer: normally the envelope should just return the data
            # directly, but older versions of Torchserve didn't have envelope.
            image = row.get("data") or row.get("body") or row.get("file")  # BugBox adds uploaded images on the 'file' field
            if isinstance(image, str):
                # if the image is a string of bytesarray.
                image = base64.b64decode(image)

            # If the image is sent as bytesarray
            if isinstance(image, (bytearray, bytes)):
                image = Image.open(io.BytesIO(image))
                image = self.image_processing(image)
            else:
                # if the image is a list
                image = torch.FloatTensor(image)

            images.append(image)

        return torch.stack(images).to(self.device)

    def image_processing(self, image):
        """
        Process image to be fed to the model
        Args:
            image: PIL image

        Returns: Transformed image
        """

        # Check if image is in the correct format and colorspace
        if isinstance(image, Image.Image):
            image = image.convert('RGB')
        else:
            raise TypeError(f'Expected a PIL image but got {type(image)} instead')

        # Transform
        image = self.transform_image(image)

        return image

    @torch.no_grad()
    def infer(self, images):
        """
        Get top 3 predictions on a batch of images
        Args:
            images: Batch of images of shape [N, C, W, H]

        Returns: Tensor of shape [N, 3] of top 3 predictions for every image in the batch
        """

        predictions = self.model(images)
        probabilities = self.softmax(predictions)
        probabilities, class_indexes = torch.topk(probabilities, 3, dim=1)

        output = {'class_indices': class_indexes,
                  'probabilities': probabilities}

        return output

    def postprocess(self, inference_data, minimum_confidence=0.0):
        """
        Generate response dictionary. The response follows BugBox's required format, i.e.:

        [{ "taxonid": 5108951,
          "confidence": "99.92",
          "labels": "Diphthera festiva",
          "optional_preds": [
            {
              "pred_op": "0.07",
              "class_op": "Pantala flavescens"
            },
            {
              "pred_op": "0.00",
              "class_op": "Marimatha nigrofimbria"}]

        Args:
            inference_data: Tensor of top 3 prediction probabilities (the output of `infer` method)
            minimum_confidence: Minimum confidence to set the classification, otherwise is set to 'Unclassified'

        Returns: List of response dictionary

        """

        predictions = torch.sort(inference_data['probabilities'], dim=1, descending=True)

        prediction_probabilities = predictions.values.tolist()
        prediction_indices = predictions.indices.tolist()
        class_indices = inference_data['class_indices'].tolist()

        output = []

        for confidences, ranks, indices in zip(prediction_probabilities, prediction_indices, class_indices):
            primary_confidence, primary_class = confidences[0], indices[ranks[0]]

            if primary_confidence < minimum_confidence:
                taxonid = 0
                labels = 'Unclassified'
            else:
                taxonid = self.taxons[primary_class]['taxon_id']
                labels = self.taxons[primary_class]['class_name']

            response = {'taxonid': int(taxonid),
                        'confidence': round(primary_confidence*100, 2),  # Convert to percentage
                        'labels': labels,
                        'optional_preds': []}

            # Add optional predictions if there is a confident primary prediction
            if primary_confidence > minimum_confidence:
                for optional_confidence, optional_rank in zip(confidences[1:], ranks[1:]):
                    optional_class = indices[optional_rank]
                    response['optional_preds'].append({
                        'pred_op': round(optional_confidence*100, 2),
                        'class_op': self.taxons[optional_class]['class_name']
                    })

            output.append(response)

        return output

    def handle(self, data, context):
        """
        Entrypoint method for all prediction requests
        Args:
            data: Input data for prediction
            context: Initial context contains model server system properties.

        Returns: Output response
        """
        images = self.preprocess(data)
        inference = self.infer(images)
        output = self.postprocess(inference)

        return output
