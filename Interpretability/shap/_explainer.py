"""
===============================================================================
shap_explainer.py

explainer class for singlemodal or multimodal inputs
===============================================================================
"""
import matplotlib.pyplot as plt
import shap
import numpy as np
import copy
from . import _utils as utils
from mmf.models.mmbt import MMBT
from transformers import AutoTokenizer
from PIL import Image
from typing import List


class Explainer(object):
    """ Use shap as explainer for classification models for image, text or both.
    """

    def __init__(self, model: "MMF Model", algorithm="partition",
                 max_evals=None, batch_size=None,
                 tokenizer=None
                 ):
        """ Initialise the explianer object 

        Args:
            model: mmf model, or any model that has a .classify method as the prediction 
                method. It should take a PIL image object and a string to give the classification output

            algorithm: currently support ("partition", )

            max_evals: maximum evaluation time, default 200 if not given
            batch_size: default 50

            tokenizer: used for text input, default pretrianed
                distilbert-base-uncased-finetuned-sst-2-english from AutoTokenizer

        """
        self._supported_algos = ("partition", )
        self._supported_modes = ("multimodal", "text_only", "image_only")

        # input validations
        if algorithm not in self._supported_algos:
            raise ValueError(f"This algotithm {algorithm} is not supported!")

        # model should have a .classify method
        if not hasattr(model, 'classify'):
            raise ValueError(f"Model object must have a .classify attribute.")

        # public features
        self.model = model
        self.algorithm = algorithm
        # some methods may allow speeding up
        self.max_evals = max_evals if max_evals is not None else 200
        self.batch_size = batch_size if batch_size is not None else 50

        # internal features
        self._tokenizer = AutoTokenizer.from_pretrained(
            'distilbert-base-uncased-finetuned-sst-2-english', use_fast=True)
        self._fixed_image = None
        self._fixed_text = None
        self._tokens = None

    def _f_multimodal(self, img_txt: np.ndarray):
        """ Multimodal funtion for shap to explain

        Args:
            img_txt: np.ndarray of shape (N, ...)
                N = number of samples
                ... = shape of images with extra row for text

        Returns:
            model outputs for those samples

        """
        out = np.zeros((len(img_txt), 2))  # output same shape
        # seperate image array with text
        images, texts = self._images_texts_split(
            img_txt, self._tokens, self._tokenizer)

        # inputs neeeds to be [PIL.Image.Image]; if not try to transform
        if not isinstance(images[0], Image.Image):
            images = utils.arr_to_img(images)
        # DEBUG USE
        print(f"f_mm(), {texts=}")
        return self._fill_predictions(out, images, texts)

    def _f_image(self, images: np.ndarray):
        """ Image-only function for shap to explain

        Args:
            images: np.ndarray of shape (N, D1, D2, C);
                N = number of samples
                D1, D2, C = three channel image

        Returns:
            numpy array of predictions with shape = (N, 2);
                - N[i] = score of the image being i
        """
        out = np.zeros((len(images), 2))  # output same shape

        # inputs neeeds to be [PIL.Image.Image]; if not try to transform
        if not isinstance(images[0], Image.Image):
            images = utils.arr_to_img(images)

        texts = [self._fixed_text for _ in range(len(images))]

        return self._fill_predictions(out, images, texts)

    def _f_text(self, texts: np.ndarray):
        """ Text-only function for shap to explain
        testing only texts
        Args:
            texts: np.ndarray[str] of shape (N,)
                N = number of samples

        Returns:
            numpy array of predictions with shape = (N, 2);
                - N[i] = score of the image being i
        """
        out = np.zeros((len(texts), 2))  # output same shape
        images = [self._fixed_image for _ in range(len(texts))]

        return self._fill_predictions(out, images, texts)

    def explain(self, images: np.ndarray, texts: np.ndarray, mode: str):
        """ Main API to calculate shap values

        Args:
            images: np.ndarray of shape (N, D1, D2, C);
                N = number of samples
                D1, D2, C = three channel image
            texts: np.ndarray of shape (N,)
            mode: ("text_only", "image_only", "multimodal")

        Returns:
            a list of shap values calculated
            a tuple of (text_shap_values, image_shap_values) is returned if mode
            is "multimodal"

        """

        # input validations
        if mode not in self._supported_modes:
            raise ValueError(f"This mode {mode} is not supported!")

        if images.shape[0] != texts.shape[0]:
            raise ValueError(
                f"Shape mismatch, inputs' first dimensions should be equal!")

        # output list
        shap_values = []

        if mode == "text_only":
            if not isinstance(images[0], Image.Image):
                images = utils.arr_to_img(images)
            # initialise masker and explainer
            text_masker = shap.maskers.Text(self._tokenizer)
            # NOTE: if using text heatmap need to have output_names arg in .Explainer()
            explainer = shap.Explainer(
                self._f_text, text_masker, algorithm=self.algorithm)

            # loop through samples
            for i in range(len(images)):
                self._fixed_image = images[i]
                values = explainer(
                    texts[i:i + 1], max_evals=self.max_evals, batch_size=self.batch_size)
                shap_values.append(values)

        elif mode == "image_only":
            # initialise masker and explainer
            image_masker = shap.maskers.Image("inpaint_telea", images[0].shape)
            image_explainer = shap.Explainer(
                self._f_image, image_masker, algorithm=self.algorithm)

            # loop through samples
            for i in range(len(texts)):
                self._fixed_text = texts[i]
                values = image_explainer(
                    images[i:i + 1], max_evals=self.max_evals, batch_size=self.batch_size)
                shap_values.append(values)

        elif mode == "multimodal":
            # img_txt and tokens for all N samples given
            all_img_txt, all_tokens = self._combine_images_texts(
                images, texts, self._tokenizer)

            image_masker = shap.maskers.Image(
                "inpaint_telea", all_img_txt[0].shape)
            explainer = shap.Explainer(
                self._f_multimodal, image_masker, algorithm=self.algorithm)

            # loop through samples
            image_shap_values = []
            text_shap_values = []
            for i in range(len(all_img_txt)):
                self._tokens = all_tokens[i]
                shap_values = explainer(
                    all_img_txt[i:i + 1], max_evals=self.max_evals, batch_size=self.batch_size)
                img_values, txt_values = self._process_mm_shap_values(
                    shap_values, self._tokens)
                image_shap_values.append(img_values)
                text_shap_values.append(txt_values)

            # build explanation objects - NOTE: deprecated in new api as we are returning lists
            # image_shap_values = self._concat_shap_values(image_shap_values)
            # text_shap_values = self._concat_shap_values(text_shap_values)

            # return tuple if multimodal
            return image_shap_values, text_shap_values

        # return single-modal outputs
        return shap_values

    @staticmethod
    def image_plot(shap_values: List):
        """ plot the image chart given shap values

            Args:
                shap_values: list of shap values

            Returns:
                a list of figures
        """
        shap_values = copy.deepcopy(shap_values)
        figs = []
        for value in shap_values:
            value.data = value.data.astype(np.float64)
            shap.image_plot(value, show=False)
            figs.append(plt.gcf())
        return figs

    @staticmethod
    def parse_text_values(shap_values: List, label_index: int = 0):
        """ plot the image chart given shap values

            Args:
                shap_values: list of shap values
                label_index: which label the values correspond to, default 0

            Returns:
                a list of dictionarys, each containing (word: shap_value) pairs
                the shap values are relative to "base_value" which is also in the dictionary 
        """
        out = []
        for value in shap_values:
            value = value[0, ..., label_index]
            dic = {k: v for k, v in zip(value.data, value.values)}
            dic['base_value'] = value.base_values
            out.append(dic)
        return out

    # ===== helper functions =====

    def _fill_predictions(self, out: np.ndarray, images: List, texts: List):
        """ utility function used in the _f_xxx functions 

            Args:
                out: np.ndarray of zeros
                images: [PIL.Image] ready to be used in model.classify
                texts: [str] ready to be used in model.classify
                NOTE: len(out) == len(images) == len(texts)

            Returns:
                filled output with predictions 
        """
        for i, (text, image) in enumerate(zip(texts, images)):
            # classify, output is a tupe (index, score)
            ind, score = self.model.classify(image, text).values()
            out[i][ind] = score
            out[i][1 - ind] = 1 - score
        return out

    @staticmethod
    def _combine_images_texts(images, texts, tokenizer):
        """ Combines images and texts into an array

        Args:
            images: np.ndarray of shape = (N, ...)
            texts: np.ndarray[str] of shape = (N, )
                N = number of samples
                ... = dimensions of the images

        Returns:
            a tuple of np.ndarrays (img_txt, tokens)
            img_txt = array where each image is concatenated with text
            tokens = list of tokens, each element in the list is a token list for a string
        """
        assert len(images.shape) == 4, "Shape of images should be (N, D1, D2, C)"
        assert texts.shape[0] == images.shape[0], "Shape mismatch between images and texts"
        # calculate row dimension to append
        y_len = images[0].shape[1]
        z_len = images[0].shape[2]
        img_txt = []
        tokens = []
        for image, text in zip(images, texts):
            txt_ids = tokenizer.encode(text, add_special_tokens=False)
            txt_tokens = tokenizer.tokenize(text, add_special_tokens=False)
            new_row = np.zeros((1, y_len, z_len))
            new_row[0, :len(txt_tokens), 0] = txt_ids
            img_txt.append(np.concatenate([image, new_row], axis=0))
            tokens.append(txt_tokens)

        return np.array(img_txt), tokens

    @staticmethod
    def _images_texts_split(img_txt: np.ndarray, tokens: List, tokenizer):
        """ split concatenated image_text arrays up into images and texts arrays

        This function will be used in the _f_multimodal function which takes in 
        a img_txt of shape (N, ...) where the N here is the generated masked samples 
        that shap uses to compute average for 1 image and 1 text given from the user.
        Therefore the tokens here only should correspond to 1 text input. This is also
        reflected on why in the multimodal case we have to explain 1 example at a time

        Args:
            img_txt: np.ndarray where each element is an image array with corresponding text appended
                shape[0] = number of samples
            tokens: list of tokens
            tokenizer: used to encode and decode the tokens

        Returns:
            tuple of (list[image_array], list[texts])
        """

        images_shape = list(img_txt.shape)
        images_shape[1] -= 1  # deleting a row which is for text
        images = []
        texts = []
        # for all samples
        for i in range(img_txt.shape[0]):
            images.append(img_txt[i, :-1, ...].copy())
            text_arr = img_txt[i, -1, :len(tokens), 0].astype(int)
            texts.append(tokenizer.decode(text_arr, skip_special_tokens=True))
        return images, texts

    @staticmethod
    def _process_mm_shap_values(shap_values: shap.Explanation, tokens: List):
        """ Split multimodal shapley values
        """
        image_values = shap_values[:, :-1]
        text_values = shap_values[:, -1, :len(tokens), 0]
        text_values.data = np.array(tokens)[np.newaxis, :]
        return image_values, text_values

    @staticmethod
    def _concat_shap_values(shap_values: List):
        """ Build an explanation object with all shapley values concatenated
        """
        values = np.concatenate([s.values for s in shap_values], axis=0)
        data = np.concatenate([s.data for s in shap_values], axis=0)
        base_values = np.concatenate(
            [s.base_values for s in shap_values], axis=0)
        clustering = np.concatenate([s.clustering for s in shap_values],
                                    axis=0) if shap_values[0].clustering is not None else None
        hierarchical = np.concatenate([s.hierarchical_values for s in shap_values],
                                      axis=0) if shap_values[0].hierarchical_values is not None else None

        return shap.Explanation(
            values=values,
            base_values=base_values,
            data=data,
            clustering=clustering,
            hierarchical_values=hierarchical
        )


def _examples():
    """ Example for how to use this explainer
    """
    # read data to try
    data_path = r"hm-data/"
    labels = utils.read_labels(data_path + "train.jsonl", True)
    ids = [5643]
    target_labels = [l for l in labels if l['id'] in ids]
    print(f"{target_labels = }")
    target_images, target_texts = utils.parse_labels(
        target_labels, img_to_array=True, separate_outputs=True)

    # model to explain
    model = MMBT.from_pretrained("mmbt.hateful_memes.images")

    # Explainer hyper params
    max_evals = 100
    batch_size = 50

    # test default partition algo
    explainer = Explainer(model, max_evals=max_evals, batch_size=batch_size)
    text_shap_values = explainer.explain(
        target_images, target_texts, "text_only")
    image_shap_values = explainer.explain(
        target_images, target_texts, "image_only")
    img_values, txt_values = explainer.explain(
        target_images, target_texts, mode="multimodal")

    # plots
    # explainer.text_plot(text_shap_values)
    # explainer.image_plot(image_shap_values)


if __name__ == "__main__":
    _examples()
