from PIL import Image
from .lime_multimodal import *


# prediction using mock classification model object
def multi_predict(imgs, txts):
    inputs = zip(imgs, txts)
    res = np.zeros((len(imgs), 2))

    for i, this_input in enumerate(inputs):

        res[i][0] = np.random.uniform(0, 1)
        res[i][1] = np.random.uniform(0, 1)

    return res


# prepare image and text for the explanation generation pipeline
img_path = "../img/gun.jpg"
img_try = Image.open(img_path)
text = "How I want to say hello to Asian people"
image_numpy = np.array(img_try)


def test_explainer_init():
    try:
        exp1 = LimeMultimodalExplainer(image_numpy, "How I want to say hello to Asian people")
    except:
        assert False, "Cannot initialise"
    else:
        assert True


# used multiple times, instantiate a global object to reduce computation
exp1 = LimeMultimodalExplainer(image_numpy, "How I want to say hello to Asian people")
num_sample = 50
num_sample_2 = 15


def test_data_labels():
    # use number of samples value % 10 != 0
    data, labels, distances, n_txt_features, n_img_features, \
    segments, domain_mapper = exp1.data_labels(num_sample_2, multi_predict)
    n_features = n_img_features + n_txt_features
    assert data.shape == (num_sample + 1, n_features)
    assert labels.shape == (num_sample + 1, 2)
    assert segments.shape
    assert domain_mapper
    assert distances.shape == (num_sample + 1,)

    # use number of samples value % 10 == 0
    data, labels, distances, n_txt_features, n_img_features, \
        segments, domain_mapper = exp1.data_labels(num_sample, multi_predict)

    n_features = n_img_features + n_txt_features
    assert data.shape == (num_sample + 1, n_features)
    assert labels.shape == (num_sample + 1, 2)
    assert segments.shape
    assert domain_mapper
    assert distances.shape == (num_sample + 1,)


# used multiple times, generate a global object
data, labels, distances, n_txt_features, n_img_features, \
        segments, domain_mapper = exp1.data_labels(num_sample, multi_predict)
n_features = n_img_features + n_txt_features


def test_data_labels():
    explanation1 = exp1.explain_instance(multi_predict, 50)
    assert explanation1

    unsorted_weights = list(explanation1.unsorted_weights[0])
    assert len(unsorted_weights) == n_features

    sorted_weights = list(explanation1.local_exp[0])
    assert len(sorted_weights) == n_features

    assert 0 <= explanation1.score[0] <= 1


# global object used in later tests
explanation1 = exp1.explain_instance(multi_predict, 50)


def test_explanation_image_case():

    # case1, positive=False, negative=False
    img, mask = explanation1.get_image_and_mask(explanation1.top_labels[1],
                                                positive_only=False,
                                                num_features=10,
                                                hide_rest=False)
    assert img.shape == image_numpy.shape
    assert mask.shape
    img_weights = explanation1.get_feature_and_weight()
    assert len(img_weights[0]) == n_img_features

    # case2, positive=True, negative=False
    img, mask = explanation1.get_image_and_mask(explanation1.top_labels[1],
                                                positive_only=True,
                                                num_features=10,
                                                hide_rest=True)
    assert img.shape == image_numpy.shape
    assert mask.shape
    img_weights = explanation1.get_feature_and_weight()
    assert len(img_weights[0]) == n_img_features

    # case3, positive=False, negative=True
    img, mask = explanation1.get_image_and_mask(explanation1.top_labels[1],
                                                positive_only=False,
                                                negative_only=True,
                                                num_features=10,
                                                hide_rest=True)
    assert img.shape == image_numpy.shape
    assert mask.shape
    img_weights = explanation1.get_feature_and_weight()
    assert len(img_weights[0]) == n_img_features


def test_explanation_text():
    ans = explanation1.as_list()
    assert len(ans) == n_txt_features

# coverage run -m pytest test_lime_multimodal.py
# coverage report -m lime_multimodal.py
