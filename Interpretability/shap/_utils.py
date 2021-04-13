
"""
===============================================================================
utils.py

various utilities for the hateful memes dataset
===============================================================================
"""

import os
import json
from typing import List, Tuple, Dict
import numpy as np
import cv2
from PIL import Image


def read_labels(label_file: str, append_dir: False) -> list:
    """ utility that reads data labels

    Args:
        label_file: path to jsonl file that contains the metadata
        append_dir: whether to append the directory path to 'img' element of each line

    Returns:
        list of dictionaries, each element is one json line

    """
    dir_name = os.path.dirname(label_file) + \
        '/' if append_dir == True else None
    with open(label_file, 'r') as f:
        json_lines = f.read().splitlines()
    labels = []
    for js in json_lines:
        dic = json.loads(js)
        if dir_name is not None:
            dic['img'] = dir_name + dic['img']
        labels.append(dic)
    return labels


def parse_labels(labels: List[Dict], img_to_array=False, separate_outputs=False):
    """ loads whats in the labels into a list of tuples (for now)
    WARNING: VERY MEMORY INTENSIVE FOR MANY IMAGES 

    Args:
        labels: a list of dictionary where each element is one json line that has
            'img': str, path to the image
            'text': str, caption to the image
            size: N
        img_to_array: whether to change image to its numpy array representation
            if False will be an PIL image object
        saparate_outputs: whether to separate images and texts

    Returns:
        list of tuples, each contain (img_i, caption_i); img_i could be numpy array
        or PIL object, caption_i will be string

    """
    out = []
    images = []
    texts = []
    for i, lb in enumerate(labels):
        img = Image.open(lb['img'])
        if img_to_array == True:
            img = np.array(img, dtype=np.uint8)
            if len(img.shape) == 2:
                img = img[..., np.newaxis]
        txt = lb['text']
        if separate_outputs == True:
            texts.append(txt)
            images.append(img)
        else:
            out.append((img, txt))
    if separate_outputs == True:
        return np.array(images), np.array(texts)
    else:
        return np.array(out)


def feature_preparation(model: 'MMF_MODEL', data) -> List[Tuple]:
    """ turns data into a list of tu

    Args:
        model: for now, a mmf model that has text and image processors in its
            .processor_dict method
        data: array-like that the first element is an PIL image object
            and the second is the 

    Returns:
        list of tuples with features transformed ready for downstream
    """

    txt_processor = model.processor_dict["text_processor"]
    img_processor = model.processor_dict["image_processor"]
    images = []
    texts = []
    for pair in data:
        print(pair[0])
        image = img_processor(pair[0])
        text = txt_processor({"text": pair[1]})["text"]
        images.append(image)
        texts.append(text)

    return images, texts


def resize_image(images: np.ndarray, dsize: Tuple):
    """ Resize all images to the size given

    Args:
        images: numpy array of shape (N, D1, D2, C)
                N = numper of samples
                D1 and D2 are image dimensions
                C = number of channels
        dsize: tuple to resize D1 and D2 to

    Returns:
        Transformed image array
    """
    out = []
    for img in images:
        resized = cv2.resize(
            img, dsize=dsize, interpolation=cv2.INTER_CUBIC)
        out.append(resized)
    return np.array(out)


def show_image(image: np.array):
    img = Image.fromarray(image.astype(np.uint8))
    img.show()


def arr_to_img(images):
    # greyscale to rgb
    if len(images[0].shape) == 3 and images[0].shape[2] == 1:
        images = [Image.fromarray(img.astype(np.uint8).squeeze(2), 'L')
                  for img in images]
    else:
        images = [Image.fromarray(img.astype(np.uint8)) for img in images]

    return images
