from flask import Flask, render_template, request, url_for, redirect, session
from lime4mmf import lime_mmf
from shap4mmf import shap_mmf
from mmxai.interpretability.torchray.extremal_perturbation import torchray_mmf
from mmxai.text_removal.smart_text_removal import SmartTextRemover
from datetime import timedelta,datetime
from file_manage import *
from flask_apscheduler import APScheduler
from config import APSchedulerJobConfig
from flask_sqlalchemy import SQLAlchemy
from operator import and_


app = Flask(__name__)
app.config.from_object(APSchedulerJobConfig)
app.config['SECRET_KEY'] = os.urandom(24)
#app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(seconds=120)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.secret_key = 'Secret Key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///user_info.sqlite3'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
# 查询时会显示原始SQL语句
app.config['SQLALCHEMY_ECHO'] = True
db = SQLAlchemy(app)


class user_info(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    file_name =db.Column(db.String(20))
    ip_addr = db.Column(db.String(20))
    expired_time = db.Column(db.DateTime)

    def __init__(self,file_name,ip_addr,expired_time):
        self.file_name = file_name
        self.ip_addr = ip_addr
        self.expired_time = expired_time

@app.before_request
def before_request():
    if session.get('user') is None:
        ip_addr = request.remote_addr
        users_delete = user_info.query.filter_by(ip_addr=ip_addr).all()
        if len(users_delete) != 0:
            for user_delete in users_delete:
                print(user_delete.expired_time)
                db.session.delete(user_delete)
                db.session.commit()
        user_id = generate_random_str(8)
        mkdir('./static/user/' + user_id)
        print(user_id + "created")
        session['user'] = user_id
        file_name = session['user']
        expired_time = datetime.now() + timedelta(days=1)
        user_insert = user_info(file_name, ip_addr, expired_time)
        db.session.add(user_insert)
        db.session.commit()
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
    error_exp = session.get('error')
    if img_name is None:
        img_name = 'logo.png'
    if img_exp is None:
        img_exp = 'logo.png'
    return render_template('explainers/hateful-memes.html', imgName=img_name, imgTexts=img_text, imgExp=img_exp,
                           textExp=text_exp, error=error_exp)


@app.route('/uploadImage', methods=['POST'])
def upload_image():
    file = request.files['inputImg']
    img_name = 'user/' + session['user'] + '/' + file.filename
    img_path = 'static/' + img_name
    file.save(img_path)
    session['imgName'] = img_name
    session['error'] = 'upload error'
    session['imgText'] = None
    session['imgExp'] = None
    session['textExp'] = None
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


@app.route('/explainers/hateful-memes/predict', methods=['POST'])
def predict():
    img_text = request.form['texts']
    exp_method = request.form['expMethod']
    exp_direction = request.form['expDir']
    user_model = session.get('userModel')
    print(user_model)
    img_name = session.get('imgName')
    print(img_name)
    model_type = session.get('modelType')
    model_path = session.get('modelPath')

    if exp_method == 'shap':
        text_exp, img_exp = shap_mmf.shap_multimodal_explain(
            img_name, img_text, user_model, model_type, model_path)
    elif exp_method == 'lime':
        text_exp, img_exp = lime_mmf.lime_multimodal_explain(
            img_name, img_text, user_model, model_type, model_path)
    elif exp_method == 'torchray':
        text_exp, img_exp = torchray_mmf.torchray_multimodal_explain(
            img_name, img_text, user_model, model_type, model_path)
    else:
        text_exp, img_exp = shap_mmf.shap_multimodal_explain(
            img_name, img_text, user_model, model_type, model_path)

    session['imgText'] = img_text
    session['textExp'] = text_exp
    session['imgExp'] = img_exp

    print(session['imgText'])
    print(session['textExp'])
    print(session['imgExp'])
    print(session['modelPath'])

    return redirect(url_for('hateful_memes'))


if __name__ == '__main__':
    app.run()

scheduler = APScheduler()
scheduler.init_app(app)
scheduler.start()