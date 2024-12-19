import base64
import io
import logging
from pathlib import Path

import torch
from PIL import Image

from ts.torch_handler.vision_handler import VisionHandler

from inference import MetaformerInferencer, load_mapping

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class MetaformerHandler(VisionHandler):
    """Torchserve handler for the MetaFormer model """

    def __init__(self):
        super(MetaformerHandler, self).__init__()
        self.device = 'cpu'
        self.mapping = load_mapping(Path('taxon_map.csv'))
        self.config = None
        self.checkpoint = None

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
        self.checkpoint = model_dir / serialized_file
        self.config = Path('config.yaml')

    def initialize(self, context):
        """
        Initialize model. This method is only called when a new worker is added to this model in TorchServe
        Args:
            context: Initial context contains model server system properties, this is Torchserve specific.
        """

        self.get_context(context)

        self.model = MetaformerInferencer(self.device)
        self.model.build(self.config, self.checkpoint, output_function='softmax')

        self.initialized = True

    def preprocess(self, data):
        """The preprocess function converts the input data to a float tensor
        Args:
            data (List): Input data from the request is in the form of a Tensor
        Returns:
            list : The preprocess function returns the input image as a list of float tensors.
        """
        image = None
        for row in data:
            # Compat layer: normally the envelope should just return the data
            # directly, but older versions of Torchserve didn't have envelope.
            image = row.get("data") or row.get("body") or row.get("file")  # BugBox puts uploaded images on the 'file' field
            if isinstance(image, str):
                # if the image is a string of bytesarray.
                image = base64.b64decode(image)

            # If the image is sent as bytesarray
            if isinstance(image, (bytearray, bytes)):
                image = Image.open(io.BytesIO(image))

        return image

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

            # taxonid depricated. Currently it is returning the first specimen_id for the class from the taxon_map.csv
            # instead of taxon_map.csv, a smaller, 2 column file could be created, similar to stats.csv.
            if primary_confidence >= minimum_confidence:
                labels = class_names[primary_class_index]  # Map index to id
                morphospecies, taxonid = self.mapping.get(labels, 0)  # Map id to GBIF taxon id and morphospecies name
            else:
                labels = 3458
                morphospecies = 'incertae sedis'
                taxonid = 0

            response = {'confidence': round(primary_confidence*100, 2),  # Convert to percentage
                        'morphospecies_id': int(labels),
                        'name': morphospecies,
                        'optional_preds': [],
                        'modelVersion': self.manifest['model']['modelVersion']}

            # Add optional predictions if there is a confident primary prediction
            if primary_confidence >= minimum_confidence:
                for optional_confidence, optional_class_index in zip(confidences[1:], indices[1:]):
                    optional_name, optional_id = self.mapping.get(class_names[optional_class_index], 0)
                    response['optional_preds'].append({
                        'pred_op': round(optional_confidence*100, 2),
                        'class_op': optional_name
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
        predictions = self.model(images)
        output = self.postprocess(predictions)

        return output
