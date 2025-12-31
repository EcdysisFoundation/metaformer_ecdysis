import io
import logging
from pathlib import Path
from abc import ABC

import torch
from PIL import Image

from ..torch_handler.base_handler import BaseHandler

from inference import MetaformerInferencer, load_mapping

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class MetaformerHandler(BaseHandler, ABC):
    """Torchserve handler for the MetaFormer model """

    def __init__(self):
        # TODO: replace paths with your path location
        super(MetaformerHandler, self).__init__()
        self.device = 'cpu'
        self.mapping = load_mapping(Path('morphospecies_map.csv'))
        self.checkpoint = Path('best.pth')
        self.config = Path('config.yaml')

    def initialize(self):
        """
        Initialize model. In Torchserve, this method is called when a new worker is added to this model.
        """
        self.model = MetaformerInferencer(self.device)
        self.model.build(self.config, self.checkpoint, output_function='softmax')
        self.initialized = True

    def preprocess(self, image_data):

        try:
            img = Image.open(io.BytesIO(image_data))
            return img.convert("RGB")

        except Exception:
            return None

    def postprocess(self, inference_data, minimum_confidence=0.0):
        """
        Generate response dictionary. The response follows BugBox's required format, i.e.:

        [{
          "confidence": "99.92",
          "morphospecies_id": 3458,
          "name": "Diphthera festiva",
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
        class_names = self.model.config.DATA.CLASS_NAMES

        probabilities, class_indices = torch.topk(inference_data, 3, dim=1, sorted=True)

        output = []

        for confidences, indices in zip(probabilities.tolist(), class_indices.tolist()):
            primary_confidence, primary_class_index = confidences[0], indices[0]

            if primary_confidence >= minimum_confidence:
                labels = class_names[primary_class_index]  # Map index to id
                morphospecies = self.mapping.get(labels, 0)  # Map id morphospecies name
            else:
                labels = 3458
                morphospecies = 'incertae sedis'

            response = {'confidence': round(primary_confidence*100, 2),  # Convert to percentage
                        'morphospecies_id': int(labels),
                        'name': morphospecies,
                        'optional_preds': [],
                        'modelVersion': self.model.config.VERSION
                        }

            # Add optional predictions if there is a confident primary prediction
            if primary_confidence >= minimum_confidence:
                for optional_confidence, optional_class_index in zip(confidences[1:], indices[1:]):
                    optional_name = self.mapping.get(class_names[optional_class_index], 0)
                    response['optional_preds'].append({
                        'pred_op': round(optional_confidence*100, 2),
                        'class_op': optional_name
                    })

            output.append(response)

        return output

    def handle(self, image_data):
        """
        Entrypoint method for all prediction requests
        Args:
            data: io.By

        Returns: Output response
        """

        imgbytes = self.preprocess(image_data)
        if not imgbytes:
            return [{'message': 'Waring: PIL could not open image'}]
        predictions = self.model(imgbytes)
        output = self.postprocess(predictions)

        return output
