from flask import Flask, render_template, request, url_for, redirect, session
from mmxai.interpretability.classification.lime import lime_mmf
from shap4mmf import shap_mmf
from mmxai.interpretability.classification.torchray.extremal_perturbation import torchray_mmf
from mmxai.text_removal.smart_text_removal import SmartTextRemover
from app_utils import prepare_explanation, text_visualisation
import os
import random


app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.secret_key = 'Secret Key'


def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print(path + "Directory created successfully")
        return True
    else:
        print(path + "Directory has already existed!")
        return False


def generate_random_str(randomlength=16):
    random_str = ''
    base_str = 'ABCDEFGHIGKLMNOPQRSTUVWXYZabcdefghigklmnopqrstuvwxyz0123456789'
    length = len(base_str) - 1
    for i in range(randomlength):
        random_str += base_str[random.randint(0, length)]
    return random_str


@app.before_request
def before_request():
    user_id = generate_random_str(8)
    if session.get('user') is None:
        mkdir('./static/user/' + user_id)
        print(user_id + "created")
        session['user'] = user_id
    else:
        print(session.get('user') + " has existed")


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/docs/')
def docs():
    return render_template('docsify.html')


@app.route('/explainers/hateful-memes')
def hateful_memes():
    img_name = session.get('imgName')
    img_exp = session.get('imgExp')
    img_text = session.get('imgText')
    text_exp = session.get('textExp')
    user_model = session.get('userModel')
    model_type = session.get('modelType')
    model_path = session.get('modelPath')
    cls_result = session.get('clsResult')
    exp_text_visl = session.get('textVisual')

    if img_name is None:
        img_name = 'logo.png'
    if img_exp is None:
        img_exp = 'logo.png'

    if model_path is not None:
        model_name = model_path.split('/')[-1]
        # only display part of the filename if the filename is longer than 6 chars
        name, extension = os.path.splitext(model_name)
        if len(name) > 6:
            model_name = name[0:3] + '...' + name[-1] + extension
    else:
        model_name = None

    if user_model == 'no_model':
        cur_opt = 'MMF ({}) with pretrained checkpoint'.format(model_type)
    elif user_model == 'mmf':
        cur_opt = 'MMF ({}) with user checkpoint'.format(model_type)
    elif user_model == 'onnx':
        cur_opt = 'ONNX'
    else:
        cur_opt = None

    print("xxx", exp_text_visl)
    return render_template('explainers/hateful-memes.html', imgName=img_name, imgTexts=img_text, imgExp=img_exp,
                           textExp=text_exp, clsResult=cls_result, curOption=cur_opt, fileName=model_name, textVisual=exp_text_visl)


@app.route('/uploadImage', methods=['POST'])
def upload_image():
    file = request.files['inputImg']
    img_name = 'user/' + session['user'] + '/' + file.filename
    img_path = 'static/' + img_name
    file.save(img_path)
    session['imgName'] = img_name
    session['imgText'] = None
    session['imgExp'] = None
    session['textExp'] = None
    session['textVisual'] = None
    session['clsResult'] = None
    return redirect(url_for('hateful_memes'))


@app.route('/uploadModel', methods=['POST'])
def upload_model():
    option = request.form['selectOption']
    if option == 'internalModel':
        selectedModel = request.form['selectModel']
        file = request.files['inputCheckpoint1']
        file_path = 'user/' + session['user'] + '/' + file.filename
        session['modelPath'] = file_path
        session['modelType'] = selectedModel
        session['userModel'] = 'mmf'
        file.save('static/' + file_path)
    elif option == 'selfModel':
        file = request.files['inputCheckpoint2']
        file_path = 'user/' + session['user'] + '/' + file.filename
        session['modelPath'] = file_path
        session['modelType'] = None
        session['userModel'] = 'onnx'
        file.save('static/' + file_path)
    elif option == 'noModel':
        selectedExistingModel = request.form['selectExistingModel']
        session['modelType'] = selectedExistingModel
        session['modelPath'] = None
        session['userModel'] = 'no_model'
    else:
        raise Exception("Sorry, you must select an option to continue!!!")
    return redirect(url_for('hateful_memes'))


@app.route('/inpaint')
def inpaint():
    img_name = session.get('imgName')
    # Prepare the image path and names
    folder_path = 'static/'
    image_path = folder_path + img_name

    img_name_no_extension = os.path.splitext(img_name)[0]
    img_extension = os.path.splitext(img_name)[1]

    inpainted_image_name = img_name_no_extension + "_inpainted" + img_extension
    save_path = folder_path + inpainted_image_name
    print(save_path)
    if not os.path.isfile(save_path):
        # Load the inpainter
        inpainter = SmartTextRemover(
            "../mmxai/text_removal/frozen_east_text_detection.pb")

        # Inpaint image
        img_inpainted = inpainter.inpaint(image_path)

        # save inpainted image
        img_inpainted.save(save_path)

    session['imgName'] = inpainted_image_name
    return redirect(url_for('hateful_memes'))


@app.route('/restoreImage')
def restore():
    img_path = session['imgName']
    root, extension = os.path.splitext(img_path)
    if '_inpainted' in root:
        img_path = root[:-10] + extension
        session['imgName'] = img_path
    return redirect(url_for('hateful_memes'))


@app.route('/explainers/hateful-memes/predict', methods=['POST'])
def predict():
    img_text = request.form['texts']
    exp_method = request.form['expMethod']
    exp_direction = request.form['expDir']
    user_model = session.get('userModel')
    img_name = session.get('imgName')
    model_type = session.get('modelType')
    model_path = session.get('modelPath')

    model, label_to_explain, cls_label, cls_confidence = prepare_explanation(
        img_name, img_text, user_model, model_type, model_path, exp_direction)

    hateful = 'HATEFUL' if cls_label == 1 else 'NON-HATEFUL'
    cls_result = 'Your uploaded image and text combination ' \
                 'looks like a {} meme, with {}% confidence.'.format(hateful, "%.2f" % (cls_confidence * 100))
    print(cls_result)

    if exp_method == 'shap':
        text_exp, img_exp = shap_mmf.shap_multimodal_explain(img_name, img_text, model)
    elif exp_method == 'lime':
        text_exp, img_exp = lime_mmf.lime_multimodal_explain(img_name, img_text, model)
    elif exp_method == 'torchray':
        text_exp, img_exp = torchray_mmf.torchray_multimodal_explain(img_name, img_text, model)
    else:
        text_exp, img_exp = shap_mmf.shap_multimodal_explain(img_name, img_text, model)

    session['clsResult'] = cls_result
    session['imgText'] = img_text
    session['textExp'] = text_exp
    session['imgExp'] = img_exp

    img_exp_name, _ = os.path.splitext(img_exp)
    exp_text_visl = img_exp_name + '_text.png'
    try:
        print("SHAP text")
        print(type(text_exp))
        print(text_exp)

        text_visualisation(text_exp, cls_label, exp_text_visl)
        session['textVisual'] = exp_text_visl
    except:
        print("error unable to plot shap text")
        session['textVisual'] = None

    print(session['imgText'])
    print(session['textExp'])
    print(session['imgExp'])
    print(session['modelPath'])

    return redirect(url_for('hateful_memes'))


if __name__ == '__main__':
    app.run()
