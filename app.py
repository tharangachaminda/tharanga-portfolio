from flask import Flask, render_template, request, jsonify, url_for
import json
import datetime
from dash import Dash, html, dcc, callback, Output, Input, ctx, State
import plotly.express as px
import pandas as pd
import dash_bootstrap_components as dbc
import csv
import sqlite3
import sqlalchemy
import os
import os.path
import sys
import requests
import base64

#os.environ['KERAS_BACKEND']='theano'
#import tensorflow as tf
#from tensorflow.keras.models import load_model
#from keras.preprocessing.image import load_img, img_to_array
#from keras.applications.inception_v3 import preprocess_input as preprocess_input_inception_v3

import pickle
import numpy as np
import time

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

try:
    from PIL import ImageEnhance
    from PIL import Image as pil_image
except ImportError:
    pil_image = None
    ImageEnhance = None

if pil_image is not None:
    _PIL_INTERPOLATION_METHODS = {
        'nearest': pil_image.NEAREST,
        'bilinear': pil_image.BILINEAR,
        'bicubic': pil_image.BICUBIC,
    }
    # These methods were only introduced in version 3.4.0 (2016).
    if hasattr(pil_image, 'HAMMING'):
        _PIL_INTERPOLATION_METHODS['hamming'] = pil_image.HAMMING
    if hasattr(pil_image, 'BOX'):
        _PIL_INTERPOLATION_METHODS['box'] = pil_image.BOX
    # This method is new in version 1.1.3 (2013).
    if hasattr(pil_image, 'LANCZOS'):
        _PIL_INTERPOLATION_METHODS['lanczos'] = pil_image.LANCZOS
        
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer # Document Term Frequency
import rake_nltk
from rake_nltk import Rake
import nltk
#nltk.download('stopwords')
#nltk.download('punkt')

#from ml_apps import app as machine_leanrning_apps
#from dl_apps import app as deep_learning_apps
#from recommender_system import app as recommender_system
#from dashboards.falcon9.falcon9 import app as falcon9_dashboard
#from dashboards.co2_emission.co2_dash import app as co2_emission_dashboard

app = Flask(__name__)

# ML app routes
with open('ML_models/lr_banknotes_model.pkl', 'rb') as f:
    banknote_model = pickle.load(f)

with open('ML_models/Blackfriday_DT_model.pkl', 'rb') as f:
    black_friday_dt_model = pickle.load(f)
    
with open('ML_models/heart_SVC_model.pkl', 'rb') as f:
    heart_attack_model = pickle.load(f)
    
with open('ML_models/diabetes_DT_model.pkl', 'rb') as f:
    diabetes_model = pickle.load(f)
        
my_work_f = open("my_work.json")
my_work = json.load(my_work_f)

@app.context_processor
def inject_now():
    return {'now': datetime.datetime.utcnow()}
    
@app.route("/")
def home():
    return render_template('home.html', mywork=my_work)
    
@app.route("/black_friday")
def black_friday():
    return render_template('black_friday.html', mywork=my_work)


@app.route('/black_friday_prediction', methods=['POST'])
def black_friday_prediction():
    if request.method == 'POST':
        form_data = request.get_json()

        city_category_options = np.array([1, 2, 3])
        city_category_one_hot = city_category_options == int(
            form_data['city_category'])
        city_category_one_hot = np.array(
            list(map(int, city_category_one_hot))).astype('float')

        # prepare data array
        input_data = np.array([
            form_data['gender'],
            form_data['age'],
            form_data['occupation'],
            form_data['stay_in_current_city'],
            form_data['marital_status'],
            form_data['product_main_category'],
            form_data['product_category_1'],
            form_data['product_category_2'],
        ]).astype('float')

        input_data_ = np.append(input_data, city_category_one_hot)
        input_data_ = input_data_.reshape(-1, 1)

        # scaling data using standardScaler
        std_scaler = StandardScaler()
        input_data_scalled = std_scaler.fit_transform(input_data_)

        # predict now
        input_data_final = input_data_scalled.reshape(1, -1)
        # print(input_data_final)
        pred = black_friday_dt_model.predict(input_data_final)

        return render_template('model_result.html', message={"type": 'normal', 'text': f"Predicted purchase amount for above customer: <b>${str(round(pred[0], 2))}</b>"})


@app.route('/banknotes_authentication')
def banknotes_authentication():
    return render_template('banknotes_authentication.html')


@app.route('/banknotes_auth', methods=['POST'])
def banknotes_auth():
    if request.method == 'POST':
        form_data_json = request.get_json()

        try:
            model_input = np.array([
                form_data_json['variance'],
                form_data_json['skewness'],
                form_data_json['curtosis'],
                form_data_json['entropy']]).astype('float')

            model_input = model_input.reshape(1, - 1)
            predict = banknote_model.predict(model_input)
        except:
            return render_template('model_result.html', message={'type': 'error', 'text': "Something went wrong!"})

        output_str = ["invalid", "valid"]

        return render_template('model_result.html', message={'type': 'normal', 'text': f"Model says it's a <b>{output_str[predict[0]]}</b> banknote."})


@app.route('/heart_attack_prediction', methods=['GET', 'POST'])
def heat_attak_prediction():
    if request.method == 'GET':
        return render_template('heart_attack.html')
    elif request.method == 'POST':
        form_data_json = request.get_json()
        
        dataset_ = {
            'age': [float(form_data_json['age'])], 
            'trtbps': [float(form_data_json['trtbps'])], 
            'chol': [float(form_data_json['chol'])], 
            'thalachh': [float(form_data_json['thalachh'])], 
            'oldpeak': [float(form_data_json['oldpeak'])],
            'sex_0':[0],'sex_1': [0], 
            'cp_0': [0], 'cp_1': [0], 'cp_2': [0], 'cp_3': [0], 
            'fbs_0': 0, 'fbs_1': 0, 
            'restecg_0': [0], 'restecg_1': [0], 'restecg_2': [0], 
            'exng_0': [0], 'exng_1': [0], 
            'slp_0': [0], 'slp_1': [0], 'slp_2': [0],
            'caa_0': [0], 'caa_1': [0], 'caa_2': [0], 'caa_3': [0], 'caa_4': [0], 
            'thall_0': [0], 'thall_1': [0], 'thall_2': [0], 'thall_3': [0]}
        
        dataset_['sex_' + form_data_json['sex']] = [1]
        dataset_['cp_' + form_data_json['cp']] = [1]
        dataset_['fbs_' + form_data_json['fbs']] = [1]
        dataset_['restecg_' + form_data_json['restecg']] = [1]
        dataset_['exng_' + form_data_json['exang']] = [1]
        dataset_['slp_' + form_data_json['slp']] = [1]
        dataset_['caa_' + form_data_json['caa']] = [1]
        dataset_['thall_' + form_data_json['thall']] = [1]
        
        data_frame = pd.DataFrame(dataset_)
        #print(data_frame)
        predict = heart_attack_model.predict(data_frame)
        
        output_str = ["💚 You do not have a risk of a heart attack", "❤️ You have a risk of a heart attack"]
        
        print(output_str[predict[0]])        
        return render_template('model_result.html', message={'type': 'normal', 'text': f"{output_str[predict[0]]}"})

@app.route('/diabetes_risk_prediction', methods=['GET', 'POST'])
def diabetes_risk_prediction():
    if request.method == 'GET':
        return render_template('diabetes_risk.html')
    elif request.method == 'POST':
        form_data_json = request.get_json()
        
        dataset_ = {
            "Age": [float(form_data_json['age'])],
            "Gender_Female": [0], "Gender_Male": [0],
            "Polyuria_No": [0], "Polyuria_Yes": [0],
            "Polydipsia_No": [0], "Polydipsia_Yes": [0],
            "sudden weight loss_No": [0], "sudden weight loss_Yes": [0],
            "weakness_No": [0], "weakness_Yes": [0],
            "Polyphagia_No": [0], "Polyphagia_Yes": [0],
            "Genital thrush_No": [0], "Genital thrush_Yes": [0],
            "visual blurring_No": [0], "visual blurring_Yes": [0],
            "Itching_No": [0], "Itching_Yes": [0],
            "Irritability_No": [0], "Irritability_Yes": [0],
            "delayed healing_No": [0], "delayed healing_Yes": [0],
            "partial paresis_No": [0], "partial paresis_Yes": [0],
            "muscle stiffness_No": [0], "muscle stiffness_Yes": [0],
            "Alopecia_No": [0], "Alopecia_Yes": [0],
            "Obesity_No": [0], "Obesity_Yes": [0]
        }
        
        dataset_["Gender_" + form_data_json['gender']] = [1]
        dataset_["Polyuria_" + form_data_json['plyuria']] = [1]
        dataset_["Polydipsia_" + form_data_json['polydipsia']] = [1]
        dataset_["sudden weight loss_" + form_data_json['sudden_weight_loss']] = [1]
        dataset_["weakness_" + form_data_json['weakness']] = [1]
        dataset_["Polyphagia_" + form_data_json['polyphagia']] = [1]
        dataset_["Genital thrush_" + form_data_json['genital_thrush']] = [1]
        dataset_["visual blurring_" + form_data_json['visual_blurring']] = [1]
        dataset_["Itching_" + form_data_json['itching']] = [1]
        dataset_["Irritability_" + form_data_json['irritability']] = [1]
        dataset_["delayed healing_" + form_data_json['delayed_healing']] = [1]
        dataset_["partial paresis_" + form_data_json['partial_paresis']] = [1]
        dataset_["muscle stiffness_" + form_data_json['muscle_stiffness']] = [1]
        dataset_["Alopecia_" + form_data_json['alopecia']] = [1]
        dataset_["Obesity_" + form_data_json['obesity']] = [1]
        
        df = pd.DataFrame(dataset_)
        
        predict = diabetes_model.predict(df)
        
        output_str = ["💚 You do not have a risk of a diabetes", "❤️ You have a risk of a diabetes"]
        
        print(output_str[predict[0]])        
        return render_template('model_result.html', message={'type': 'normal', 'text': f"{output_str[predict[0]]}"})        

##### Deep learning apps #####

# Tf model serving with S3 bucket
tfs_host = '127.0.0.1'
#tfs_host = '52.43.46.199'
def get_tf_serving_rest_url(model_name, host=tfs_host, port='8501', verb='predict'):
     url = "http://{0}:{1}/v1/models/{2}:{3}".format(host, port, model_name, verb)
     
     return url
     
def tr_serving_rest_request(data, url):
    payload = json.dumps({"instances": data})
    response = requests.post(url=url, data=payload)
    return response

# load saved model files
mood_class_strings = ['Sad', 'Happy']
#mood_detection_model = load_model('ML_models/mood_classifier_1701759497220')
mood_detection_model = "mood_classifier_1701759497220"

# this list is not essential. We can use model output index as the result. But for the consistancy of the program I'm using it here.
sign_language_class_strings = [0, 1, 2, 3, 4, 5]
#sign_language_model = load_model('ML_models/sign_laguange.h5')
#sign_language_model = load_model('ML_models/cnn_sign_language_detection_1701800022195')
sign_language_model = "cnn_sign_language_detection_1701800022195"

#sign_language_resnet_model = load_model('ML_models/sign_language_resnet50')
sign_language_resnet_model = "resnet_50"

#alpaca_mobilenetv2_model = load_model('ML_models/alpaca_mobile_netv2_1701759341886')
alpaca_mobilenetv2_model = "alpaca_mobile_netv2_1701759341886"

handwritten_digits_model = 'mnist_digits_model'

#image_captioning_model = "model_InceptionV3_trained"
#inception_base_model_model = "model_InceptionV3_base"

#co2_emission_lstm_model = load_model('ML_models/co2_emission_lstm_1701759497110')
co2_emission_lstm_model = "co2_emission_lstm_1701759497110"

co2_emission_scalerfile = 'ML_models/co2_emission_data/scaler.sav'
co2_emission_scaler = pickle.load(open(co2_emission_scalerfile, 'rb'))
co2_orig_csv = pd.read_csv('ML_models/co2_emission_data/year_emission.csv')

up_path = os.path.join(os.getcwd(), 'static', 'assets', 'temp')

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

my_work_f = open("my_work.json")
my_work = json.load(my_work_f)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def keras_load_img(path, grayscale=False, color_mode='rgb', target_size=None, interpolation='nearest'):
    """Loads an image into PIL format.

    # Arguments
        path: Path to image file.
        color_mode: One of "grayscale", "rgb", "rgba". Default: "rgb".
            The desired image format.
        target_size: Either `None` (default to original size)
            or tuple of ints `(img_height, img_width)`.
        interpolation: Interpolation method used to resample the image if the
            target size is different from that of the loaded image.
            Supported methods are "nearest", "bilinear", and "bicubic".
            If PIL version 1.1.3 or newer is installed, "lanczos" is also
            supported. If PIL version 3.4.0 or newer is installed, "box" and
            "hamming" are also supported. By default, "nearest" is used.

    # Returns
        A PIL Image instance.

    # Raises
        ImportError: if PIL is not available.
        ValueError: if interpolation method is not supported.
    """
    if grayscale is True:
        warnings.warn('grayscale is deprecated. Please use '
                      'color_mode = "grayscale"')
        color_mode = 'grayscale'
    if pil_image is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `array_to_img` requires PIL.')
    img = pil_image.open(path)
    if color_mode == 'grayscale':
        if img.mode != 'L':
            img = img.convert('L')
    elif color_mode == 'rgba':
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
    elif color_mode == 'rgb':
        if img.mode != 'RGB':
            img = img.convert('RGB')
    else:
        raise ValueError('color_mode must be "grayscale", "rgb", or "rgba"')
    if target_size is not None:
        width_height_tuple = (target_size[1], target_size[0])
        if img.size != width_height_tuple:
            if interpolation not in _PIL_INTERPOLATION_METHODS:
                raise ValueError(
                    'Invalid interpolation method {} specified. Supported '
                    'methods are {}'.format(
                        interpolation,
                        ", ".join(_PIL_INTERPOLATION_METHODS.keys())))
            resample = _PIL_INTERPOLATION_METHODS[interpolation]
            img = img.resize(width_height_tuple, resample)
    return img

def keras_img_to_array(img, data_format='channels_last', dtype='float32'):
    """Converts a PIL Image instance to a Numpy array.

    # Arguments
        img: PIL Image instance.
        data_format: Image data format,
            either "channels_first" or "channels_last".
        dtype: Dtype to use for the returned array.

    # Returns
        A 3D Numpy array.

    # Raises
        ValueError: if invalid `img` or `data_format` is passed.
    """
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format: %s' % data_format)
    # Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but original PIL image has format (width, height, channel)
    x = np.asarray(img, dtype=dtype)
    if len(x.shape) == 3:
        if data_format == 'channels_first':
            x = x.transpose(2, 0, 1)
    elif len(x.shape) == 2:
        if data_format == 'channels_first':
            x = x.reshape((1, x.shape[0], x.shape[1]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], 1))
    else:
        raise ValueError('Unsupported image shape: %s' % (x.shape,))
    return x

def imageToArray(image_path, grayscale=False, color_mode='rgb', target_size=None, noOfChannels=3, preprocessing_method=None):
    # Load the image and resize it to the desired dimensions
    # image_path = f'images/{imageName}'
    
    imageArray = None
    
    image = keras_load_img(image_path, color_mode=color_mode, target_size=target_size)
    imageArray = keras_img_to_array(image)
    imageArray = imageArray.reshape(1, target_size[0], target_size[1], noOfChannels)
    imageArray = imageArray / 255.0

    # plt.imshow(image_array)
    # plt.show()

    # print(image_array.shape)
    # Reshape the image array to match the input shape of your model
    # Assumes the input shape is (width, height, 3)
    try:
        if preprocessing_method is not None:
            imageArray = preprocessing_method(image_array)
    except Exception as error:
        print('Image exception: ', error)
        if os.path.exists(image_path):
            os.remove(image_path)
        return False

    return imageArray


@app.route("/mood_detection")
def mood_detection():
    return render_template("mood_detection.html")

@app.route("/image_captioning", methods=["GET", "POST"])
def image_captioning():
    return render_template("image_captioning.html")
    """
    if request.method == "GET":
        return render_template("image_captioning.html")
    elif request.method == "POST":
        if 'image_file' not in request.files:
            return render_template('model_result.html', message={'type': 'error', 'text': 'You have not uploaded an image.'})
        
        image = request.files['image_file']
        
        if not allowed_file(image.filename):
            return render_template('model_result.html', message={'type': 'error', 'text': 'Not a valid image file.'})
        
        inputImageWidth, inputImageHeight = 299, 299
        preprocessing_method = None
        #preprocessing_method = keras.applications.vgg16.preprocess_input
        
        file_extension = image.filename.rsplit('.', 1)[1]
        temp_filename = f'{int(time.time())}.{file_extension}'
        image.save(os.path.join(up_path, temp_filename))
        temp_file_path = os.path.join(up_path, temp_filename)
        
        upImage = load_img(temp_file_path, target_size=(299, 299))
        upImage = img_to_array(upImage)
        upImage = upImage.reshape((1, upImage.shape[0], upImage.shape[1], upImage.shape[2]))
        upImage = preprocess_input_inception_v3(upImage)
        
        #upImage = imageToArray(temp_file_path, 299, 299, preprocess_input_inception_v3)
        
        # imageArray = imageToArray(
        #     temp_file_path, inputImageWidth, inputImageHeight, preprocessing_method)

        # if imageArray is False:
        #     return render_template('model_result.html', message={'type': 'error', 'text': 'Your image has some issues.'})
        
        tf_serving_incep_url = get_tf_serving_rest_url(inception_base_model_model)
        
        image_features_response = tr_serving_rest_request(upImage.tolist(), tf_serving_incep_url)
        image_features_json = image_features_response.json()
        print('get image features: ', image_features_json)
        
        # error handling
        if 'error' in image_features_json:
            return render_template('model_result.html', message={'type': 'error', 'text': "Something went wrong!"})
        
        with open('ML_models/tokenizer.pkl', 'rb') as f:
            tokenizer = load(f) 
        
        generate_desc(tokenizer, image_features_json['predictions'], 34)
        
        return 'Works so far'
        """

def generate_desc(tokenizer, photo, max_length):
    # seed the generation process
    in_text = 'startseq'
    
    tf_serving_url = get_tf_serving_rest_url(image_captioning_model)
        
    # iterate over the whole length of the sequence
    for i in range(max_length):
        # print("output %d" % i)
        # integer encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad input
        sequence = pad_sequences([sequence], maxlen=max_length)
        # predict next word
        # yhat = model.predict([photo, sequence], verbose=0)
        tr_serving_response = tr_serving_rest_request([photo, sequence], tf_serving_url)
        tr_serving_response_json = tr_serving_response.json()
        print(tr_serving_response_json)
        
        yhat = tr_serving_response_json['predictions']
    
        # convert probability to integer
        yhat = argmax(yhat)
        
        # map integer to word
        word = word_for_id(yhat, tokenizer)
        # print("output2 %s %d" % (word, yhat))
        # stop if we cannot map the word
        if word is None:
            break
        # append as input for generating the next word
        in_text += ' ' + word
        # stop if we predict the end of the sequence
        if word == 'endseq':
            break
    return in_text        

def word_for_id(integer, tokenizer):
    """
    This function will map integer to a word in the tokenizer.

    Parameters:
        integer - Integer Word ID
        tokenizer - Keras tikenizer

    Return:
        Relevant word - If found in the tokenizer. None otherwise
    """
    for word, index in tokenizer.word_index.items():
        #print('tokenizer: %d %s' % (index, word))
        if index == integer:
            return word

    return None

@app.route("/predict_cnn/<task>", methods=["POST"])
def predict_cnn(task):
    if request.method == 'POST':
        #if 'image_file' not in request.files:
        #    return render_template('model_result.html', message={'type': 'error', 'text': 'You have not uploaded an image.'})

        model_obj = mood_detection_model
        output_class_strings = mood_class_strings
        inputImageWidth = 64
        inputImageHeight = 64
        preprocessing_method = None
        imageArray = None
        colorMode = 'rgb'
        noOfChannels = 3

        request_json = request.get_json()
                
        #image = request.files['image_file']
        imageBase64 = request_json['image_file']
        file_extension = imageBase64.split(';')[0].split('/')[1]        
        #file_extension = image.filename.rsplit('.', 1)[1]
        temp_filename = f'{int(time.time())}.{file_extension}'
        temp_file_path = os.path.join(up_path, temp_filename)
        
        with open(temp_file_path, "wb") as fh:
            fh.write(base64.b64decode(imageBase64.split('base64,')[1]))        
        
        #print(temp_file_path)
        #return render_template('model_result.html', message={'type': 'error', 'text': 'Processing.'})
        #image.save(os.path.join(up_path, temp_filename))
                
        if not allowed_file(temp_filename):
            return render_template('model_result.html', message={'type': 'error', 'text': 'Not a valid image file.'})
        
        if task == "sign_language":
            model_obj = sign_language_model
            output_class_strings = sign_language_class_strings
        elif task == 'sign_language_resnet':
            model_obj = sign_language_resnet_model
            output_class_strings = sign_language_class_strings
        elif task == 'alpaca_mobilenetv2':
            model_obj = alpaca_mobilenetv2_model
            output_class_strings = ['Not an Alpaca', 'Alpaca']
            inputImageWidth = 160
            inputImageHeight = 160
            #preprocessing_method = tf.keras.applications.mobilenet_v2.preprocess_input
        elif task == 'handwritten_digits_recognition':
            model_obj = handwritten_digits_model
            output_class_strings = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
            inputImageWidth = 28
            inputImageHeight = 28
            colorMode = 'grayscale'
            noOfChannels = 1

        
        if imageArray is None:
            imageArray = imageToArray(temp_file_path, color_mode=colorMode, target_size=(inputImageWidth, inputImageHeight), noOfChannels=noOfChannels)

        if imageArray is False:
            return render_template('model_result.html', message={'type': 'error', 'text': 'Your image has some issues.'})

        tf_serving_url = get_tf_serving_rest_url(model_obj)
        
        model_predict_prob = []
        #model_predict_prob = model_obj.predict(imageArray)
        tr_serving_response = tr_serving_rest_request(imageArray.tolist(), tf_serving_url)
        tr_serving_response_json = tr_serving_response.json()
        #print(tr_serving_response_json)
        
        # error handling
        if 'error' in tr_serving_response_json:
            return render_template('model_result.html', message={'type': 'error', 'text': "Something went wrong!"})
        
        model_predict_prob = tr_serving_response_json['predictions']
        #print(model_predict_prob)

        if task == 'sign_language' or task == 'sign_language_resnet' or task == 'handwritten_digits_recognition':
            model_predict_int = np.argmax(model_predict_prob)
        else:
            model_predict_int = int(model_predict_prob[0][0] > 0.5)

        #print(model_predict_prob[0][0], model_predict_int)
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        else:
            print('file path does not exist')

        model_final_output = output_class_strings[model_predict_int]

        return render_template('model_result.html', message={'type': 'normal', 'text': f'Model predicts as <b>"{model_final_output}"</b> with probability {model_predict_prob}'})


@app.route('/sign_language_recognition')
def sign_language_recognition():
    #model_summary = sign_language_model.to_json()
    # print(model_summary)
    return render_template("sign_language_detection.html")


@app.route('/sign_language_recognition_resnet')
def sign_language_recognition_resnet():
    return render_template('resnet.html')


@app.route('/alpaca_mobilenetv2')
def alpaca_mobilenetv2():
    return render_template('alpaca_mobilenetv2.html')

@app.route('/handwritten_digits_recognition')
def handwritten_digits_recognition():
    return render_template('handwritten_digits.html')

@app.route('/co2_emission_lstm', methods=['GET', 'POST'])
def co2_emission_lstm():
    if request.method == 'GET':
        return render_template('co2_emission_lstm.html', params={'last_year': int(co2_orig_csv['year'].iloc[-1])})
    elif request.method == 'POST':
        request_json = request.get_json()
        
        window_size = 2
        
        # validation set
        val_data = np.array([[0.42428668],
                            [0.47168195],
                            [0.49305629],
                            [0.5446335 ],
                            [0.47121728],
                            [0.5265118 ],
                            [0.49026836],
                            [0.46610604],
                            [0.50234947],
                            [0.61154451],
                            [0.67799085],
                            [0.56786648],
                            [0.71284033],
                            [0.8410864 ],
                            [0.91403795],
                            [0.98048429],
                            [0.90288613],
                            [1.        ]])
        
        x_input = val_data[-(window_size):].reshape(1, -1)
        
        temp_input = list(x_input)
        temp_input = temp_input[0].tolist() # this is the inputs for starting prediction
        
        tf_serving_url = get_tf_serving_rest_url(co2_emission_lstm_model)
        
        lst_output=[]
        n_steps=window_size
        number_of_years_to_predict = int(request_json['to_year']) - 2019
        i=0
        while(i<number_of_years_to_predict):
            
            if(len(temp_input)>n_steps):
                #print(temp_input)
                x_input=np.array(temp_input[1:])
                #print("{} Year input {}".format(i,x_input))
                x_input=x_input.reshape(1,-1)
                x_input = x_input.reshape((1, n_steps, 1))
                #print(x_input)
                #yhat = co2_emission_lstm_model.predict(x_input, verbose=0)
                tr_serving_response = tr_serving_rest_request(x_input.tolist(), tf_serving_url)
                tr_serving_response_json = tr_serving_response.json()
                
                # error handling
                if 'error' in tr_serving_response_json:
                    return render_template('model_result.html', message={'type': 'error', 'text': "Something went wrong!"})
        
                yhat = tr_serving_response_json['predictions']
                #print("{} Year output {}".format(i,yhat))
                temp_input.extend(yhat[0])
                temp_input=temp_input[1:]
                #print(temp_input)
                lst_output.extend(yhat)
                i=i+1
            else:
                x_input = x_input.reshape((1, n_steps,1))
                #yhat = co2_emission_lstm_model.predict(x_input, verbose=0)
                tr_serving_response = tr_serving_rest_request(x_input.tolist(), tf_serving_url)
                tr_serving_response_json = tr_serving_response.json()
                
                # error handling
                if 'error' in tr_serving_response_json:
                    return render_template('model_result.html', message={'type': 'error', 'text': "Something went wrong!"})
                
                yhat = tr_serving_response_json['predictions']
                #print(yhat[0])
                temp_input.extend(yhat[0])
                #print(len(temp_input))
                lst_output.extend(yhat)
                i=i+1
        
        final_output = co2_emission_scaler.inverse_transform(lst_output).flatten()
        x_labels = co2_orig_csv['year'].tolist() + np.arange(int(co2_orig_csv['year'].iloc[-1]) + 1, int(request_json['to_year']) + 1).tolist()
        
        chart_data_orig = list(co2_orig_csv['value']) + [0] * len(final_output)
        chart_data_predicted = [0] * (len(co2_orig_csv['value']) - 1) + list([co2_orig_csv['value'].iloc[-1]]) + list(final_output)
        
        response_json = {"x_labels": x_labels, "orig": chart_data_orig, "predicted": chart_data_predicted}
        return jsonify(response_json)

# Recommender system
recommender_system_df = pd.read_csv('ML_models/movies.csv') # https://query.data.world/s/uikepcpffyo2nhig52xxeevdialfl7
recommender_system_df_orig = recommender_system_df[['Title', 'Year', 'Genre','Director','Actors','Plot', 'Poster', 'imdbRating']].set_index('Title')
recommender_system_df = recommender_system_df[['Title', 'Year', 'Genre','Director','Actors','Plot', 'Poster', 'imdbRating']]

def recomender_preprocessing(movie_df):
    # data preprocessing for recomender system
    # set lowercase and split on commas
    movie_df['Actors'] = movie_df['Actors'].map(lambda x: x.lower().split(','))
    movie_df['Genre'] = movie_df['Genre'].map(lambda x: x.lower().split(','))
    movie_df['Director'] = movie_df['Director'].map(lambda x: x.lower().split(' '))

    # join Actors and Director names as a single string
    movie_df['Actors'] = movie_df.apply(lambda row: [x.replace(' ', '') for x in row['Actors']], axis=1)
    movie_df['Director'] = movie_df.apply(lambda row: ''.join(row['Director']), axis=1)
    # for index, row in movie_df.iterrows():
    #     #print(index)
    #     row['Actors'] = [x.replace(' ', '') for x in row['Actors']]
    #     row['Director'] = ''.join(row['Director'])
        
    #     movie_df.loc[:, ('Actors', index)] = row['Actors']
    #     movie_df.loc[:, ('Director', index)] = row['Director']
    
    return movie_df

recommender_system_df = recomender_preprocessing(recommender_system_df)

# Extract keywords using Rake()

def extract_keywords(movie_df):
    movie_df['Key_words'] = "" # initialize the column for storing keywords
    
    for index, row in movie_df.iterrows():
        plot = row['Plot']
        
        # initiate Rake
        r = Rake()
        
        # extracting the words by passing the text
        r.extract_keywords_from_text(plot)
        
        # preparing a dictionary with keywords and their scores
        keyword_dict_score = r.get_word_degrees()
        
        # assign keywords to the new column
        row['Key_words'] = list(keyword_dict_score.keys())
       
    # we do not need 'Plot' column anymore
    movie_df.drop('Plot', axis=1, inplace=True)
    
    # set Title as index, then we can easily identify records rather than using numerical indices
    movie_df.set_index('Title', inplace=True)
    
    return movie_df 

recommender_system_df = extract_keywords(recommender_system_df)

# create bag of words
def create_bag_of_words(movie_df):
    # initialize bag_of_words column
    movie_df['bag_of_words'] = ""
    
    columns = movie_df.columns
    # for index, row in movie_df.iterrows():
    #     words = ''
    #     for col in columns:
    #         if col == 'Director':
    #             #print(row[col])
    #             words += row[col] + ' '
    #         elif col == 'Actors':
    #             words += ' '.join(row[col]) + ' '
    #     movie_df.loc[:, ('bag_of_words', index)] = words
    
    movie_df['bag_of_words'] = movie_df.apply(
        lambda row: bag_of_words_row(row['Actors'], row['Director']),
        axis=1
    )
        
    # now we only need the index and the bag_of_words column. So, we drop other columns
    keep_cols = ['bag_of_words', 'Title', 'Year', 'Director', 'Poster', 'imdbRating']
    movie_df.drop([col for col in columns if col not in keep_cols], axis=1, inplace=True)
    
    return movie_df

def bag_of_words_row(actors, director):
    words = ''
    words += director + ' '
    words += ' '.join(actors) + ' '
    
    return words

recommender_system_df = create_bag_of_words(recommender_system_df)
#print(recommender_system_df[['Director', 'bag_of_words']])
#print(recommender_system_df)

# apply Countervectorizer
# this tokenize the words by counting the frequesncy. This is needed for calculate Cosine similarity
# after that in the same function cosine_similarity finction is also applied and return cosine_sim matrix
def count_vectorizer(df):
    count_vector = CountVectorizer()
    count_matrix = count_vector.fit_transform(df['bag_of_words'])
    
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    
    return cosine_sim

cosine_sim = count_vectorizer(recommender_system_df)

# implement recomender function
indeces = pd.Series(recommender_system_df.index)

def recommender(title, cosine_sim=cosine_sim):
    recommendations = []
    
    # get relevant indeces
    if len(indeces[indeces == title]) > 0:
        search_idx = indeces[indeces == title].index[0]
    else:
        return []
    
    similarities = pd.Series(cosine_sim[search_idx]).sort_values(ascending=False)
    
    # get top 10 matches (indexes)
    # use this indexes again to retrieve movie titles
    top_10_matches = list(similarities.iloc[1:11].index)
    #print(top_10_matches)
    
    # store best matched titles
    for i in top_10_matches:
        recommendations.append(indeces[i])
        
    return recommendations

@app.route("/recommender_content_based", methods=['GET', 'POST'])
def recommender_content_based():
    if request.method == 'GET':
        return render_template('recommender_content_based.html', options=recommender_system_df.index)
    elif request.method == 'POST':
        input_movie = request.get_json()
        input_movie_title = input_movie['input_movie']
        
        output_message_type = "normal"
        grid_info = [{
                    'title': input_movie_title,
                    'year': recommender_system_df.loc[input_movie_title]['Year'],
                    'director': recommender_system_df_orig.loc[input_movie_title, 'Director'],
                    'poster': recommender_system_df.loc[input_movie_title]['Poster'],
                    'imdb': recommender_system_df.loc[input_movie_title]['imdbRating']
                }]
        
        recommended_movies = recommender(input_movie_title)
                
        #print(recommended_movies)
        if len(recommended_movies) == 0:
            recommended_movies = "No result found"
            output_message_type = "info"
        else:            
            for movie in recommended_movies:
                grid_info.append({
                    'title': movie,
                    'year': recommender_system_df.loc[movie]['Year'],
                    'director': recommender_system_df_orig.loc[movie, 'Director'],
                    'poster': recommender_system_df.loc[movie]['Poster'],
                    'imdb': recommender_system_df.loc[movie]['imdbRating']
                })
        
        return render_template('info_grid.html', message={'type': output_message_type, 'text': recommended_movies, 'grid_info': grid_info})
    
# Falcon 9 Dashboard        
sys.path.append('../../')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#print(BASE_DIR)
db_path = os.path.join(BASE_DIR, "dashboards", "falcon9", "falcon9.db")

# fetch data from SQlite database abd create a dataframe
con = sqlite3.connect(db_path)
cur = con.cursor()

sql_query = "SELECT * FROM `falcon9_tbl`"

df = pd.read_sql(sql=sql_query, con=con)
df['Flight_No'] = range(1, df.shape[0] + 1)
df['Class'] = [1 if landing_status ==
               'Success' else 0 for landing_status in df['Booster_Landing']]
df['Year'] = pd.to_datetime(df['Date']).dt.year

# print(df['Class'].value_counts())

colors = {
    'background': '#111111',
    'text': '#becdce',
    'light': '#e7e7e7',
    'lineColor': '#687879',
    'gridColor': '#384d4f'
}

binary_class_palette = ['#DE3163', '#50C878']

pieChartHeight = 370

# prepare launch sites options for selectbox

def getLaunchSitesOptions():
    launchSitesOptions = [{"label": "All Launch Sites", "value": "All"}]
    for launchSite in df['Launch_Site'].unique():
        launchSitesOptions.append({"label": launchSite, "value": launchSite})

    return launchSitesOptions


def getBoosterVersions():
    boosterVersionOptions = [{'label': 'All Booster Versions', 'value': 'All'}]
    for boosterVersion in df['Version_Booster'].unique():
        boosterVersionOptions.append(
            {'label': boosterVersion, 'value': boosterVersion})

    return boosterVersionOptions


def getOrbitTypeOptions():
    orbitTypeOptions = [{'label': 'All Orbit types', 'value': 'All'}]

    for orbitType in df['Orbit'].unique():
        if orbitType is None:
            continue
        orbitTypeOptions.append({'label': orbitType, 'value': orbitType})

    return orbitTypeOptions

# Falcon 9 Analysis app
falcon_9_dash_app = Dash(external_stylesheets=[dbc.themes.SOLAR, dbc.icons.BOOTSTRAP], server=app, url_base_pathname ="/falcon9_dashboard/")
falcon_9_dash_app.title = "Falcon 9"

@falcon_9_dash_app.callback(
    Output(component_id="success_launchsite_pie_chart",
           component_property="figure"),
    Input(component_id="launch_site", component_property="value"),
    Input('flight_number', 'value'),
    Input('payload_mass', 'value'),
    Input('booster_version', 'value'),
    Input('orbit_type', 'value'),
    Input('year', 'value'),
)
def getSuccessPieChart(launch_site, flight_number, payload_mass, booster_version, orbit_type, year):
    filtered_df = df[((df['Flight_No'] >= flight_number[0]) & (df['Flight_No'] <= flight_number[1]))
                     & ((df['Payload_Mass'] >= payload_mass[0]) & (df['Payload_Mass'] <= payload_mass[1]))
                     & ((df['Year'] >= year[0]) & (df['Year'] <= year[1]))]

    chart_title = "Success landing percentage by Launch Site"

    # filter by booster versoin
    if booster_version != "All":
        filtered_df = df[df['Version_Booster']
                         == booster_version]
    # filter by Orbit type
    if orbit_type != 'All':
        filtered_df = df[df['Orbit'] == orbit_type]

    filtered_df = filtered_df[['Launch_Site', 'Class']]

    if launch_site == None or launch_site == "All":
        fig = px.pie(filtered_df,
                     values='Class',
                     names='Launch_Site',
                     title=chart_title,
                     color_discrete_sequence=px.colors.qualitative.Antique,
                     hole=.3)

    else:
        filtered_df = filtered_df[filtered_df['Launch_Site'] ==
                                  launch_site].value_counts().to_frame().reset_index()

        chart_title = "Landing outcome for Launch Site %s" % launch_site

        # set color sequence for single output value
        color_sequence = binary_class_palette

        if len(filtered_df['Class']) > 0:
            if len(filtered_df['Class']) == 1:
                color_sequence = [
                    binary_class_palette[filtered_df['Class'][0]]]
            else:
                color_sequence = [binary_class_palette[filtered_df['Class']
                                                       [0]], binary_class_palette[filtered_df['Class'][1]]]

        filtered_df['Class'] = filtered_df['Class'].replace(
            {0: 'Failure', 1: 'Success'})

        # print(filtered_df)
        fig = px.pie(filtered_df,
                     values='count',
                     names='Class',
                     title=chart_title,
                     color_discrete_sequence=color_sequence,
                     hole=.3,)

    fig = updateChartLayout(filtered_df, fig, chart_title, pieChartHeight)

    return fig


@falcon_9_dash_app.callback(
    Output('success_orbit_pie_chart', 'figure'),
    Input('orbit_type', 'value'),
    Input("launch_site", "value"),
    Input('flight_number', 'value'),
    Input('payload_mass', 'value'),
    Input('booster_version', 'value'),
    Input('year', 'value'),
)
def getSuccessOrbitPieChart(orbit_type, launch_site, flight_number, payload_mass, booster_version, year):
    filtered_df = df[~df['Orbit'].isnull()]
    filtered_df = df[((df['Flight_No'] >= flight_number[0]) & (df['Flight_No'] <= flight_number[1]))
                     & ((df['Payload_Mass'] >= payload_mass[0]) & (df['Payload_Mass'] <= payload_mass[1]))
                     & ((df['Year'] >= year[0]) & (df['Year'] <= year[1]))]

    # filter by Booster Version
    if booster_version != 'All':
        filtered_df = df[df['Version_Booster'] == booster_version]

    # filter by Launch Site
    if launch_site != None and launch_site != 'All':
        filtered_df = df[df['Launch_Site'] == launch_site]

    filtered_df['Orbit'] = filtered_df['Orbit'].replace(
        'Ballistic lunar transfer (BLT)', 'BLT')

    filtered_df = filtered_df[['Orbit', 'Class']]

    chart_title = "Success landing percentage for all Orbit types"

    if orbit_type == 'All':
        fig = px.pie(
            filtered_df,
            values='Class',
            names='Orbit',
            color_discrete_sequence=px.colors.qualitative.Antique,
            hole=.3)
    else:
        filtered_df = filtered_df[filtered_df['Orbit'] == orbit_type
                                  ].value_counts().to_frame().reset_index()

        chart_title = "Landing Outcome for Orbit %s" % orbit_type

        # set color sequence for single output value
        color_sequence = binary_class_palette

        if len(filtered_df['Class']) > 0:
            if len(filtered_df['Class']) == 1:
                color_sequence = [
                    binary_class_palette[filtered_df['Class'][0]]]
            else:
                color_sequence = [binary_class_palette[filtered_df['Class']
                                                       [0]], binary_class_palette[filtered_df['Class'][1]]]

        filtered_df['Class'] = filtered_df['Class'].replace(
            {0: 'Failure', 1: 'Success'}, inplace=True)

        fig = px.pie(filtered_df,
                     values='count',
                     names='Class',
                     title=chart_title,
                     color_discrete_sequence=color_sequence,
                     hole=.3,)
    #print('len df 2: \n', filtered_df)
    fig = updateChartLayout(filtered_df, fig, chart_title, pieChartHeight)

    return fig


@falcon_9_dash_app.callback(
    Output('success_boosterversion_pie_chart', 'figure'),
    Input(component_id="launch_site", component_property="value"),
    Input('flight_number', 'value'),
    Input('payload_mass', 'value'),
    Input('booster_version', 'value'),
    Input('orbit_type', 'value'),
    Input('year', 'value'),
)
def getSuccessRateBoosterVersionPieChart(launch_site, flight_number, payload_mass,  booster_version, orbit_type, year):
    filtered_df = df[((df['Flight_No'] >= flight_number[0]) & (df['Flight_No'] <= flight_number[1]))
                     & ((df['Payload_Mass'] >= payload_mass[0]) & (df['Payload_Mass'] <= payload_mass[1]))
                     & ((df['Year'] >= year[0]) & (df['Year'] <= year[1]))]

    # filrer by Launch Site
    if launch_site != None and launch_site != 'All':
        filtered_df = filtered_df[filtered_df['Launch_Site'] == launch_site]

    # filter by Orbit type
    if orbit_type != 'All':
        filtered_df = df[df['Orbit'] == orbit_type]

    filtered_df = filtered_df[['Version_Booster', 'Class']]

    chart_title = "Successful landings by Booster Version"

    if booster_version == 'All':
        fig = px.pie(
            filtered_df,
            values='Class',
            names='Version_Booster',
            title=chart_title,
            color_discrete_sequence=px.colors.qualitative.Antique,
            hole=.3)
    else:
        filtered_df = filtered_df[filtered_df['Version_Booster'] ==
                                  booster_version].value_counts().to_frame().reset_index()

        # set color sequence for single output value
        color_sequence = binary_class_palette

        if len(filtered_df['Class']) > 0:
            if len(filtered_df['Class']) == 1:
                color_sequence = [
                    binary_class_palette[filtered_df['Class'][0]]]
            else:
                color_sequence = [binary_class_palette[filtered_df['Class']
                                                       [0]], binary_class_palette[filtered_df['Class'][1]]]

        chart_title = "Landing outcome for Booster Version %s" % booster_version

        filtered_df['Class'] = filtered_df['Class'].replace(
            {0: 'Failure', 1: 'Success'})

        fig = px.pie(filtered_df,
                     values='count',
                     names='Class',
                     color_discrete_sequence=color_sequence,
                     hole=.3,)

    fig = updateChartLayout(filtered_df, fig, chart_title, pieChartHeight)

    return fig


@falcon_9_dash_app.callback(
    Output(component_id='launch_site_v_payload_mass',
           component_property='figure'),
    Input(component_id="launch_site", component_property="value"),
    Input('flight_number', 'value'),
    Input('payload_mass', 'value'),
    Input('booster_version', 'value'),
    Input('orbit_type', 'value'),
    Input('year', 'value'),
)
def getLaunchSiteVsPayloadMass(launch_site, flight_number, payload_mass, booster_version, orbit_type, year):
    filtered_df = df[((df['Flight_No'] >= flight_number[0]) & (df['Flight_No'] <= flight_number[1]))
                     & ((df['Payload_Mass'] >= payload_mass[0]) & (df['Payload_Mass'] <= payload_mass[1]))
                     & ((df['Year'] >= year[0]) & (df['Year'] <= year[1]))]

    chart_title = "Landing outcome of Launch Site for against Payload Mass"

    # filter by booster versoin
    if booster_version != "All":
        filtered_df = df[df['Version_Booster']
                         == booster_version]

    # filter by Orbit type
    if orbit_type != 'All':
        filtered_df = df[df['Orbit'] == orbit_type]

    if launch_site == None or launch_site == "All":
        fig = px.scatter(filtered_df, x='Payload_Mass', y='Class', color='Launch_Site',
                         labels={
                             "Payload_Mass": "Payload Mass (kg)",
                             "Launch_Site": "Launch Site",
                             "Class": "Landing Outcome"
                         },
                         color_discrete_sequence=px.colors.qualitative.Dark2,
                         category_orders={'Class': ['Fail', 'Success']}
                         )
    else:
        filtered_df = filtered_df[filtered_df['Launch_Site'] == launch_site]
        fig = px.scatter(filtered_df, x='Payload_Mass', y='Class', color='Launch_Site',
                         labels={
                             "Payload_Mass": "Payload Mass (kg)",
                             "Launch_Site": "Launch Site",
                             "Class": "Landing Outcome"
                         },
                         color_discrete_sequence=px.colors.qualitative.Dark2,
                         category_orders={'Class': ['Fail', 'Success']}
                         )

        chart_title = "Landing outcome of Launch Site %s against Payload Mass" % (
            launch_site)

    fig.update_xaxes(zeroline=False, linecolor=colors['lineColor'],
                     gridcolor=colors['gridColor'], tickvals=[0, 1])
    fig.update_yaxes(linecolor=colors['lineColor'], type='category',
                     gridcolor=colors['gridColor'])

    fig = updateChartLayout(filtered_df, fig, chart_title, 250)

    fig.update_layout(
        paper_bgcolor='rgba(0, 0, 0, 0.2)',
    )

    return fig


# YEARLY TREND CHARTS
@falcon_9_dash_app.callback(
    Output('success_yearly_trend_linechart', 'figure'),
    Input('year', 'value'),
    Input("launch_site", "value"),
    Input('flight_number', 'value'),
    Input('payload_mass', 'value'),
    Input('booster_version', 'value'),
    Input('orbit_type', 'value')
)
def getYearlySuccessTrendLineChart(year, launch_site, flight_number, payload_mass,  booster_version, orbit_type):
    filtered_df = df[((df['Flight_No'] >= flight_number[0]) & (df['Flight_No'] <= flight_number[1]))
                     & ((df['Payload_Mass'] >= payload_mass[0]) & (df['Payload_Mass'] <= payload_mass[1]))
                     & ((df['Year'] >= year[0]) & (df['Year'] <= year[1]))]

    # filrer by Launch Site
    if launch_site != None and launch_site != 'All':
        filtered_df = filtered_df[filtered_df['Launch_Site'] == launch_site]

    # filter by Orbit type
    if orbit_type != 'All':
        filtered_df = df[df['Orbit'] == orbit_type]

    # filter by Booster Version
    if booster_version != 'All':
        filtered_df = df[df['Version_Booster'] == booster_version]

    chart_title = "Success Landing Yearly Trend %s - %s" % (year[0], year[1])

    filtered_df_line = filtered_df[['Year', 'Class']
                              ].groupby('Year').sum(numeric_only=True).reset_index()
    
    # print(filtered_df)
    fig = px.line(
        filtered_df_line,
        x='Year',
        y='Class',
        color_discrete_sequence=['#50C878'],
        labels={
            'Class': 'Successful Landings'
        }
    )

    fig.update_yaxes(
        zeroline=False, linecolor=colors['lineColor'], 
        gridcolor=colors['gridColor'])
    fig.update_xaxes(zeroline=False, linecolor=colors['lineColor'], type='date',
                     gridcolor=colors['gridColor'])

    fig = updateChartLayout(filtered_df, fig, chart_title, 300)

    fig.update_layout(
        paper_bgcolor='rgba(0, 0, 0, 0.2)',
    )

    return fig


@falcon_9_dash_app.callback(
    Output('yearly_launches_barchart', 'figure'),
    Input('year', 'value'),
    Input("launch_site", "value"),
    Input('flight_number', 'value'),
    Input('payload_mass', 'value'),
    Input('booster_version', 'value'),
    Input('orbit_type', 'value')
)
def getYearlyLaunchesBarchart(year, launch_site, flight_number, payload_mass,  booster_version, orbit_type):
    filtered_df = df[((df['Flight_No'] >= flight_number[0]) & (df['Flight_No'] <= flight_number[1]))
                     & ((df['Payload_Mass'] >= payload_mass[0]) & (df['Payload_Mass'] <= payload_mass[1]))
                     & ((df['Year'] >= year[0]) & (df['Year'] <= year[1]))]

    # filrer by Launch Site
    if launch_site != None and launch_site != 'All':
        filtered_df = filtered_df[filtered_df['Launch_Site'] == launch_site]

    # filter by Orbit type
    if orbit_type != 'All':
        filtered_df = df[df['Orbit'] == orbit_type]

    # filter by Booster Version
    if booster_version != 'All':
        filtered_df = df[df['Version_Booster'] == booster_version]

    chart_title = "Yearly Launches %s - %s" % (year[0], year[1])

    filtered_df_bar = filtered_df[['Year']].value_counts().reset_index().sort_values(by="Year", ascending=True)
    
    filtered_df_line = filtered_df[['Year', 'Class']
                              ].groupby('Year').sum(numeric_only=True).sort_values(by="Year", ascending=True).reset_index()

    df_combined = filtered_df_bar.merge(filtered_df_line, on="Year")
    
    #print(df_combined)
    fig = px.bar(
        df_combined,
        x='Year',
        y='count',
        text_auto='.2s',
        color_discrete_sequence=[px.colors.qualitative.Prism[3]],
        labels={
            'count': 'Number of Launches'
        }
    ).add_traces(
        px.line(
            df_combined,
            x='Year',
            y='Class',
            color_discrete_sequence=['#FFBF00'],
            labels={
                'Class': 'Successful Landings'
            }
        ).update_traces(showlegend=True, name="success").data
    )

    fig.update_yaxes(
        zeroline=False, linecolor=colors['lineColor'], gridcolor=colors['gridColor'])
    fig.update_xaxes(zeroline=False, linecolor=colors['lineColor'], type='date',
                     gridcolor=colors['gridColor'], categoryorder='category ascending')

    fig = updateChartLayout(filtered_df, fig, chart_title, 300)
    fig.update_layout(
        paper_bgcolor='rgba(0, 0, 0, 0.2)',
    )

    return fig


@falcon_9_dash_app.callback(
    Output('launch_site', 'value'),
    Output('flight_number', 'value'),
    Output('payload_mass', 'value'),
    Output('booster_version', 'value'),
    Output('orbit_type', 'value'),
    Output('year', 'value'),
    Input('reset_button', 'n_clicks'),
    State('launch_site', 'value')
)
def resetFilters(nclicks, launch_site):
    return 'All', [df['Flight_No'].min(), df['Flight_No'].max()], [df['Payload_Mass'].min(), df['Payload_Mass'].max()], 'All', 'All', [df['Year'].min(), df['Year'].max()]


def updateChartLayout(filtered_df, fig, chart_title, height):
    if len(filtered_df) == 0:
        fig.update_layout(
            plot_bgcolor='rgba(0, 0, 0, 0)',
            paper_bgcolor='rgba(0, 0, 0, 0)',
            font_color=colors['text'],
            title_text=chart_title,
            title_x=0.5,
            autosize=False,
            height=height,
            xaxis={"visible": False},
            yaxis={"visible": False},
            annotations=[
                {
                    "text": "No data available for this graph.",
                    "xref": "paper",
                    "yref": "paper",
                    "showarrow": False,
                    "font": {
                        "size": 14
                    }
                }
            ]
        )

    else:
        fig.update_layout(
            plot_bgcolor='rgba(0, 0, 0, 0)',
            paper_bgcolor='rgba(0, 0, 0, 0)',
            font_color=colors['text'],
            title_text=chart_title,
            title_x=0.5,
            autosize=False,
            height=height
        )

    return fig

falcon_9_dash_app.layout = html.Div(
    [
        dbc.Container(
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        html.Div(
                                            html.Img(
                                                src="/static/assets/images/Falcon_9_logo.png",
                                                width=100,
                                                style={
                                                    'margin-bottom': '20px',
                                                })
                                        ),
                                        width=3
                                    ),

                                    dbc.Col(
                                        html.Div(
                                            [
                                                html.H2("SpaceX Falcon 9", style={
                                                        'color': colors['light']}),
                                                html.H5(
                                                    "Stage 1 Landing Analysis", style={
                                                        'color': colors['light']})
                                            ]
                                        ),
                                        width=9,
                                        align='center',
                                    )

                                ]
                            ),

                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.Div(
                                                [
                                                    dbc.Button(
                                                        [
                                                            html.I(className="bi bi-house-fill me-2"),
                                                            "Back to Home"
                                                        ], 
                                                        href="javascript:history.back()",
                                                        id='back_button', 
                                                        color="warning", 
                                                        className="me-1", 
                                                        n_clicks=0)
                                                ]
                                            ),
                                            
                                            html.Hr(),
                                            
                                            dbc.Label("Launch Site"),
                                            dcc.Dropdown(
                                                id='launch_site',
                                                options=[{'label': ls, 'value': ls}
                                                         for ls in df['Launch_Site'].unique()] + [{'label': 'All Launch Sites', 'value': 'All'}],
                                                value="All",
                                                placeholder="Select Launch Site",
                                                style={
                                                    'background': colors['text'],
                                                    'color': '#333333',
                                                    'border-radius': '4px',
                                                }
                                            ),

                                            html.Hr(),

                                            dbc.Label("Flight Number"),
                                            dcc.RangeSlider(
                                                id="flight_number",
                                                min=0, max=df['Flight_No'].max() + 1, step=25,
                                                value=[df['Flight_No'].min(),
                                                       df['Flight_No'].max()],
                                                allowCross=False,
                                                tooltip={
                                                    "placement": "bottom", "always_visible": False},
                                                marks={fn: {'label': str(fn), 'style': {'color': str(
                                                    colors['text'])}} for fn in list(range(0, df['Flight_No'].max() + 1, 25))}
                                            ),

                                            html.Hr(),

                                            dbc.Label("Payload Mass (kg)"),
                                            dcc.RangeSlider(
                                                id="payload_mass",
                                                min=0, max=df['Payload_Mass'].max() + 1, step=1000,
                                                value=[df['Payload_Mass'].min(),
                                                       df['Payload_Mass'].max()],
                                                allowCross=False,
                                                tooltip={
                                                    "placement": "bottom", "always_visible": False},
                                                marks={fn: {'label': str(fn), 'style': {'color': str(
                                                    colors['text'])}} for fn in list(range(0, round(df['Payload_Mass'].max()) + 1, 2000))}
                                            ),

                                            html.Hr(),

                                            dbc.Label("Booster Version"),
                                            dcc.Dropdown(
                                                id="booster_version",
                                                options=[{'label': bv, 'value': bv}
                                                         for bv in df['Version_Booster'].unique()] + [{'label': 'All Booster Versions', 'value': 'All'}],
                                                value="All",
                                                searchable=True,
                                                placeholder="Select a Booster Version",
                                                style={
                                                    'background': colors['text'],
                                                    'color': '#333333',
                                                    'border-radius': '4px',
                                                }
                                            ),

                                            html.Hr(),

                                            dbc.Label("Orbit"),
                                            dcc.Dropdown(
                                                id="orbit_type",
                                                options=[{'label': orbit, 'value': orbit}
                                                         for orbit in df['Orbit'].unique()] + [{'label': 'All Orbit types', 'value': 'All'}],
                                                value="All",
                                                searchable=True,
                                                placeholder="Select an Orbit type",
                                                style={
                                                    'background': colors['text'],
                                                    'color': '#333333',
                                                    'border-radius': '4px',
                                                }
                                            ),

                                            html.Hr(),

                                            dbc.Label("Year"),
                                            dcc.RangeSlider(
                                                id="year",
                                                min=df['Year'].min(), max=df['Year'].max(), step=1,
                                                allowCross=False,
                                                tooltip={
                                                    "placement": "bottom", "always_visible": False},
                                                marks={
                                                    yr: {'label': str(yr), 'style': {
                                                        'color': str(colors['text'])}}
                                                    for yr in list(range(df['Year'].min(), df['Year'].max() + 1, 2))
                                                },
                                                value=[df['Year'].min(),
                                                       df['Year'].max()],
                                            ),

                                            html.Hr(),

                                            dbc.Button(
                                                "Reset Filters", id='reset_button', color="primary", className="me-1", n_clicks=0)

                                        ]
                                    ),
                                ]
                            )

                        ],

                        width=3),

                    dbc.Col(

                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        html.Div(
                                            dcc.Graph(
                                                id="success_launchsite_pie_chart")
                                        ),
                                        width=4
                                    ),

                                    dbc.Col(
                                        html.Div(
                                            dcc.Graph(
                                                id="success_boosterversion_pie_chart")
                                        ),
                                        width=4
                                    ),

                                    dbc.Col(
                                        html.Div(
                                            dcc.Graph(
                                                id="success_orbit_pie_chart")
                                        ),
                                        width=4
                                    )
                                ]
                            ),

                            dbc.Row(
                                [
                                    dbc.Col(
                                        html.Div(
                                            dcc.Graph(
                                                id="launch_site_v_payload_mass")
                                        ),
                                    )
                                ],

                                className='mb-2'
                            ),

                            dbc.Row(
                                [
                                    dbc.Col(
                                        html.Div(
                                            dcc.Graph(
                                                id="success_yearly_trend_linechart")
                                        ),
                                        width=6
                                    ),

                                    dbc.Col(
                                        html.Div(
                                            dcc.Graph(
                                                id="yearly_launches_barchart")
                                        ),
                                        width=6
                                    )
                                ]
                            )

                        ],

                        width=9)
                ]
            ),
            fluid=True
        )

    ],
    style={
        'padding': 10,
        'color': colors['text'],
    }
)

@app.route('/falcon9_dashboard')
def falcon9_dash():
    return falcon_9_dash_app.index()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def os_path(filename):
    return os.path.join(BASE_DIR, filename)

colors = {
    'background': '#111111',
    'text': '#becdce',
    'light': '#e7e7e7',
    'dark': '#515151',
    'lineColor': '#687879',
    'gridColor': '#384d4f',
    'br_gridColor': '#454545'
}

def updateChartLayout(filtered_df, fig, chart_title, height):
    if len(filtered_df) == 0:
        fig.update_layout(
            plot_bgcolor='rgba(0, 0, 0, 0)',
            paper_bgcolor='rgba(0, 0, 0, 0)',
            font_color=colors['text'],
            title_text=chart_title,
            title_x=0.5,
            autosize=False,
            height=height,
            xaxis={"visible": False},
            yaxis={"visible": False},
            annotations=[
                {
                    "text": "No data available for this graph.",
                    "xref": "paper",
                    "yref": "paper",
                    "showarrow": False,
                    "font": {
                        "size": 14
                    }
                }
            ]
        )

    else:
        fig.update_layout(
            plot_bgcolor='rgba(0, 0, 0, 0)',
            paper_bgcolor='rgba(0, 0, 0, 0)',
            font_color=colors['text'],
            title_text=chart_title,
            title_x=0.5,
            autosize=False,
            height=height
        )

    return fig

# CO2 emission Analysis app
co2_emission_dash_app = Dash(external_stylesheets=[dbc.themes.DARKLY, dbc.icons.BOOTSTRAP], server=app, url_base_pathname ="/co2_emission_dashboard/")
co2_emission_dash_app.title = "CO2 emission"

co2_all_countries = pd.read_csv(os_path('dashboards/co2_emission/data/all_countries.csv'))
co2_geo_region_df = pd.read_csv(os_path('dashboards/co2_emission/data/geo_regions.csv'))
co2_economy_region_df = pd.read_csv(os_path('dashboards/co2_emission/data/economy_groups.csv'))

geo_json_r = open(os_path('dashboards/co2_emission/data/world-countries.json'), 'r')
geo_json = geo_json_r.read()

@co2_emission_dash_app.callback(
    Output('co2_map_div', 'figure'),
    Input('co2_year', 'value')
)
def co2_emissionr_map(co2_year):
    year_to_display = co2_year[1]
    chart_title = f"Carbon dioxide emission by country in {year_to_display} (kilotons)"
    df_all_countries_filter_by_year = co2_all_countries[co2_all_countries['year'] == year_to_display]
    #df_all_countries_filter_by_year

    fig = px.choropleth(df_all_countries_filter_by_year, locations="country_code",
                    color="value", # lifeExp is a column of gapminder
                    hover_name="country_name", # column to add to hover information
                    color_continuous_scale=px.colors.sequential.Reds,
                    animation_frame="year",
                    labels={
                        'value': 'CO₂',
                        'country_name': 'Country',
                        'country_code': 'Country',
                        'year': 'Year'
                    })
    
    fig = updateChartLayout(df_all_countries_filter_by_year, fig, chart_title, 375)
    fig.update_yaxes(
        zeroline=False, linecolor=colors['lineColor'], gridcolor=colors['br_gridColor'])
    fig.update_geos(
        resolution=50, showocean=True, oceancolor="#222",
        showcountries=True, countrycolor="RebeccaPurple"
    )
    fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)')
    
    return fig

@co2_emission_dash_app.callback(
    Output('co2_top_5_contributors_bar_chart', 'figure'),
    Input('co2_year', 'value')
)
def top_5_contributors(year_range):
   
    df_top_5 = co2_all_countries[((co2_all_countries['year'] >= year_range[0]) & (co2_all_countries['year'] <= year_range[1]))].groupby('country_name').sum(numeric_only=True).sort_values(by='value', ascending=False).head(5)
    
    chart_title = f"Top 5 contributors to CO₂ emission ({year_range[0]}-{year_range[1]})"
    
    fig = px.bar(
            df_top_5, 
            x=df_top_5.index, 
            y='value', 
            text_auto='.2s',
            color='value',
            color_continuous_scale=px.colors.sequential.Brwnyl,
            labels={
                'value': 'CO₂ (kilotons)',
                'country_name': 'Country'
            })
    
    fig = updateChartLayout(df_top_5, fig, chart_title, 375)
    fig.update_yaxes(
        zeroline=False, linecolor=colors['lineColor'], gridcolor=colors['br_gridColor'])
    fig.update_xaxes(zeroline=False, linecolor=colors['lineColor'],
                     gridcolor=colors['br_gridColor'])
    
    fig.update_layout(
        paper_bgcolor='rgba(0, 0, 0, 0.2)',
    )
    
    return fig

@co2_emission_dash_app.callback(
    Output('co2_geo_region_line_chart', 'figure'),
    Input('co2_year', 'value')
)
def co2_geo_region_line_graph(year_range):
    
    df_filtered = co2_geo_region_df[((co2_geo_region_df['year'] >= year_range[0]) & (co2_geo_region_df['year'] <= year_range[1]))]
    
    chart_title = f"Different geographical regions between {year_range[0]} and {year_range[1]}"
    
    if (year_range[1] - year_range[0]) > 5:
        fig = px.line(
            df_filtered, 
            x='year', 
            y='value',
            color='country_name',
            labels={
                'value': 'CO₂ emission (kilotons)',
                'year': 'Year',
                'country_name': 'Geographical region'
            },
            color_discrete_sequence=px.colors.qualitative.Bold)
        
        fig.update_yaxes(
        zeroline=False, linecolor=colors['lineColor'], gridcolor=colors['br_gridColor'])
        fig.update_xaxes(zeroline=False, linecolor=colors['lineColor'], 
                        gridcolor=colors['br_gridColor'], type='date', categoryorder='category ascending')
    else:
        fig = px.bar(
            df_filtered, 
            x='value', 
            y='year', 
            orientation='h',
            text_auto='.2s',
            color='country_name',
            color_discrete_sequence=px.colors.qualitative.Antique,
            labels={
                'value': 'CO₂ (kilotons)',
                'country_name': 'Geographical region'
            })
        
        fig.update_yaxes(
        zeroline=False, linecolor=colors['lineColor'], gridcolor=colors['br_gridColor'], type='date', categoryorder='category ascending')
        fig.update_xaxes(zeroline=False, linecolor=colors['lineColor'], 
                        gridcolor=colors['br_gridColor'])
    
    fig = updateChartLayout(df_filtered, fig, chart_title, 375)
            
    fig.update_layout(
        paper_bgcolor='rgba(0, 0, 0, 0.2)',
    )
    
    return fig

@co2_emission_dash_app.callback(
    Output('co2_top_5_contributors_all_time_pie_chart', 'figure'),
    Input('co2_year', 'value')
)
def top_5_contributors_all_time(year_range):
    df_total_co2 = co2_all_countries.groupby('country_name').sum(numeric_only=True).sort_values(by='value', ascending=False).head(5).reset_index()
    #print(df_total_co2)
    chart_title = "Total CO₂ emission by top contributors all time"
    
    fig = px.pie(
            df_total_co2,
            values='value',
            names='country_name',
            hole=0.3,
            color_discrete_sequence=px.colors.qualitative.Antique,
            labels={
                'value': 'CO₂ (kilotons)',
                'country_name': 'Country'
            }
        )
    
    fig = updateChartLayout(df_total_co2, fig, chart_title, 375)
    
    fig.update_layout(autosize=False)
        
    return fig

@co2_emission_dash_app.callback(
    Output('co2_economy_region_line_chart', 'figure'),
    Input('co2_year', 'value')
)
def co2_economy_region(year_range):
    
    chart_title = f"Economy groups between {year_range[0]} and {year_range[1]}"
    
    df_filtered = co2_economy_region_df[((co2_economy_region_df['year'] >= year_range[0]) & (co2_economy_region_df['year'] <= year_range[1]))]
    
    if (year_range[1] - year_range[0]) > 5:
        fig = px.line(
            df_filtered, 
            x='year', 
            y='value',
            color='country_name',
            labels={
                'value': 'CO₂ emission (kilotons)',
                'year': 'Year',
                'country_name': 'Economy group'
            },
            color_discrete_sequence=px.colors.qualitative.Vivid)
    else:
        fig = px.bar(
            df_filtered, 
            x='year', 
            y='value', 
            text_auto='.2s',
            color='country_name',
            color_discrete_sequence=px.colors.sequential.Brwnyl_r,
            labels={
                'value': 'CO₂ (kilotons)',
                'country_name': 'Economy group'
            })
    
    fig = updateChartLayout(df_filtered, fig, chart_title, 375)
    
    fig.update_yaxes(
        zeroline=False, linecolor=colors['lineColor'], gridcolor=colors['br_gridColor'])
    fig.update_xaxes(zeroline=False, linecolor=colors['lineColor'], 
                     gridcolor=colors['br_gridColor'], type='date', categoryorder='category ascending')  
        
    fig.update_layout(
        paper_bgcolor='rgba(0, 0, 0, 0.2)',
    )
    
    return fig

co2_emission_dash_app.layout = html.Div(
    [
        dbc.Container(
            html.Div(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                html.Div(
                                    [
                                        html.Img(
                                            src="/static/assets/images/co2_dash_logo.png",
                                            width=120)                                           
                                    ]
                                ),
                                width='auto'
                            ),
                        
                            dbc.Col(
                                html.Div(
                                    [
                                        html.H2(
                                            [
                                                "CO₂ Emission ",
                                                html.Small('(Worldwide Analysis)', style={
                                                    'font-size': '24px'
                                                })
                                            ], style={
                                                'color': colors['text'],
                                                'display': 'inline'
                                            }
                                        ),
                                    ]
                                ),
                                width='auto'
                            ),
                            dbc.Col(
                                html.Div(
                                    [
                                        dbc.Button(
                                            [
                                                html.I(className="bi bi-house-fill me-2"),
                                                "Back to Home"
                                            ], 
                                            href="javascript:history.back()",
                                            id='back_button', 
                                            color="danger", 
                                            className="me-1", 
                                            n_clicks=0)
                                    ],
                                    style={'text-align': 'right'}
                                ),
                            )
                        ],
                        className='mb-2',
                        align="center"
                    ),
                    html.Hr(),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Div(
                                        [
                                            dcc.Graph(id="co2_map_div"),
                                        ]                          
                                    ),
                                ],
                                className="col col-md-12 col-lg-12 col-xl-5"
                            ),
                            dbc.Col(
                                [
                                    html.Div(
                                        [
                                            dcc.Graph(id="co2_top_5_contributors_bar_chart")
                                        ]                           
                                    ),
                                ],
                                className="col col-md-6 col-lg-6 col-xl-4"
                            ),
                            dbc.Col(
                                [
                                    html.Div(
                                        [
                                            dcc.Graph(id="co2_top_5_contributors_all_time_pie_chart")
                                        ]                           
                                    ),
                                ],
                                className="col col-md-6 col-lg-6 col-xl-3"
                            )
                        ],
                        className='mb-2'
                    ), 
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Div(
                                        [
                                            dcc.Graph(id="co2_geo_region_line_chart")
                                        ]                           
                                    ),
                                ],
                                className="col col-md-12 col-lg-6"
                            ),
                            dbc.Col(
                                [
                                    html.Div(
                                        [
                                            dcc.Graph(id="co2_economy_region_line_chart")
                                        ]                           
                                    ),                                    
                                ],
                                className="col col-md-12 col-lg-6"
                            )
                        ],
                        className='mb-3'
                    ), 
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.H6([
                                        html.I(className="bi bi-sliders me-2"),
                                        'Timeline'
                                        ], style={'text-align': 'center'}),
                                    dcc.RangeSlider(
                                        id="co2_year",
                                        min=co2_all_countries['year'].min(), max=co2_all_countries['year'].max(), step=1,
                                        tooltip={"placement": "top", "always_visible": False},
                                        marks={
                                            yr: {'label': str(yr), 'style': {
                                                'color': str(colors['text']), 'font-size': '16px', 'padding-bottom': 35}}
                                            for yr in list(range(co2_all_countries['year'].unique().min(), co2_all_countries['year'].unique().max() + 1, 10))
                                        },
                                        value=[co2_all_countries['year'].unique().min(), co2_all_countries['year'].unique().max()],
                                        allowCross=False
                                    )
                                ]
                            )
                        ],
                        style={'position': 'fixed', 'width': '100%', 'bottom': '0px', 'background': '#222222'}
                    )                   
                ]
            ),
            fluid=True
        ),
        
    ],
    style={
        'padding': 10,
        'padding-bottom': 50,
        'color': colors['text'],
    }
)

@app.route('/co2_emission_dashboard')
def co2_emission_dash():
    return co2_emission_dash_app.index()

@app.route('/medbot')
def medbot():
    return render_template('medbot.html') 
# application = DispatcherMiddleware(app, 
#     {
#         #"/falcon9_dashboard": falcon_9_dash_app.server,
#         #"/co2_emission_dashboard": co2_emission_dashboard.server,
#         "/machine_leanrning_apps": machine_leanrning_apps,
#         "/deep_learning_apps": deep_learning_apps,
#         "/recommender_content_based": recommender_system,
#     }
# )

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)
    #app.run(debug=True)
    #run_simple('localhost', 5000, application)
