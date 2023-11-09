from flask import Flask, render_template, request, jsonify
from PIL import Image
import os
import numpy as np
import time
import sklearn
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

import pickle
app = Flask(__name__)

mood_class_strings = ['Sad', 'Happy']
mood_detection_model = tf.keras.models.load_model(
    'ML_models/happy_model.keras')
with open('ML_models/lr_banknotes_model.pkl', 'rb') as f:
    banknote_model = pickle.load(f)

# this list is not essential. We can use model output index as the result. But for the consistancy of the program I'm using it here.
sign_language_class_strings = [0, 1, 2, 3, 4, 5]
sign_language_model = tf.keras.models.load_model(
    'ML_models/sign_laguange.keras')
up_path = os.path.join(os.getcwd(), 'static', 'assets', 'temp')

UPLOAD_FOLDER = up_path
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

my_work = [
    {
        'header': 'End-to-end',
        'application_url': 'black_friday',
        'image': 'black_friday.png',
        'title': 'Black Friday Purchase Prediction',
        'description': "This project will understand the customer purchase behaviour (specifically, purchase amount) against various products of different categories. They have shared purchase summary of various customers for selected high volume products from last month.",
        'github': 'https://github.com/tharangachaminda/Black_Friday_Purchase',
        'icons': ['python', 'jupyterlab', 'flask', 'heroku']
    },

    {
        'header': 'End-to-end',
        'application_url': 'banknotes_authentication',
        'image': 'bank_notes.png',
        'title': 'Banknotes Authentication',
        'description': "Banknote analysis refers to the examination of paper currency to determine its legitimacy and identify potential counterfeits. In this python project, I am trying to build a <b>Classification Machine Learning models</b> to predict banknotes are genuine or forged.",
        'github': 'https://github.com/tharangachaminda/banknotes_analysis',
        'icons': ['python', 'jupyterlab', 'flask', 'heroku']
    },

    {
        'header': 'Analysis',
        'image': 'falcon_9.png',
        'title': 'SpaceX Falcon 9 1st Stage Landing Prediction',
        'description': "SpaceX re-uses the Stage 1 boosters of Falcon 9 rockets. This project is an analysis for successful Stage 1 landing prediction. I have used SpaceX API and Webscraping for data collection, SQlite database for data storage.",
        'github': 'https://github.com/tharangachaminda/Falcon9_First_stage_Landing',
        'icons': ['python', 'jupyterlab', 'sql', 'sqlite', 'folium', 'plotly-dash', 'heroku']
    },

    {
        'header': 'Recommender System',
        'image': 'netflix.png',
        'title': 'Netflix Recommender System (Popularity Based)',
        'description': "Netflix Recommender system is one of the best recommender systems in the world. In this project I've used <b>movies and rating</b> datasets and <b>NLTK toolkit</b> to build this popularity based recommender system.",
        'github': 'https://github.com/tharangachaminda/popularity-based-recommendation-system',
        'icons': ['Python', 'jupyterlab', 'nltk', 'flask', 'heroku']
    },

    {
        'header': 'Recommender System',
        'image': 'netflix.png',
        'title': 'Netflix Recommender System (Content Based)',
        'description': "Netflix Recommender system is one of the best recommender systems in the world. In this project I've used <b>movies datasets</b> and <b>Cosine similarity</b> to build this content based recommender system.",
        'github': 'https://github.com/tharangachaminda/content_based_recommender_system',
        'icons': ['Python', 'jupyterlab', 'nltk', 'flask', 'heroku']
    },

    {
        'header': 'Analysis',
        'image': 'bookstore.png',
        'title': 'Bookstore Web scraping',
        'description': "In this project I have built a mechanism to collect information about every book in the website, scraping through pagination. This project is associated with https://books.toscrape.com/ website which is specially design for training web scraping. ",
        'github': 'https://github.com/tharangachaminda/bookstore_webscraping',
        'icons': ['Python', 'jupyterlab', 'flask', 'heroku']
    },

    {
        'header': 'End-to-end (Deep Learning)',
        'application_url': 'mood_detection',
        'image': 'mood_classifier.png',
        'title': 'Mood Classifier',
        'description': "A mood classification is a type of machine learning task that is used to recognize human moods or emotions. In this project I have implemented a CNN model for recognizing smiling or not smiling humans using Tensorflow Keras Sequential API.",
        'github': 'https://github.com/tharangachaminda/cnn_mood_classifier',
        'icons': ['python', 'jupyterlab', 'tensorflow', 'flask', 'heroku']
    },

    {
        'header': 'End-to-end (Deep Learning)',
        'application_url': 'sign_language_recognition',
        'image': 'sign_language_digits.png',
        'title': 'Sign Language Digits Recognition',
        'description': "Sing language is a visual-gestural language used by deaf and hard-to-hearing individuals to convey imformation, thoughts and emotions. In this project I have implemented a CNN model for recognizing sign language digits 0 to 5 using Keras Functional API.",
        'github': 'https://github.com/tharangachaminda/cnn_sign_language_detection',
        'icons': ['python', 'jupyterlab', 'tensorflow', 'flask', 'heroku']
    },

]


@app.route("/")
def home():
    return render_template('home.html', mywork=my_work)


@app.route("/mood_detection")
def mood_detection():
    model_summary = model_obj.summary()
    print(model_summary)
    return render_template("mood_detection.html", mood_model={"model_obj": model_summary})


@app.route("/predict_cnn/<task>", methods=["POST"])
def predict_cnn(task):
    if request.method == 'POST':
        if 'image_file' not in request.files:
            return render_template('model_result.html', message={'type': 'error', 'text': 'You have not uploaded an image.'})

        model_obj = mood_detection_model
        output_class_strings = mood_class_strings
        if task == "sign_language":
            model_obj = sign_language_model
            output_class_strings = sign_language_class_strings

        image = request.files['image_file']

        if not allowed_file(image.filename):
            return render_template('model_result.html', message={'type': 'error', 'text': 'Not a valid image file.'})

        file_extension = image.filename.rsplit('.', 1)[1]
        temp_filename = f'{int(time.time())}.{file_extension}'
        image.save(os.path.join(up_path, temp_filename))
        temp_file_path = os.path.join(up_path, temp_filename)
        imageArray = imageToArray(temp_file_path)

        if imageArray is False:
            return render_template('model_result.html', message={'type': 'error', 'text': 'Your image has some issues.'})

        model_predict_prob = model_obj.predict(imageArray)
        print(model_predict_prob)

        if task == 'sign_language':
            model_predict_int = np.argmax(model_predict_prob)
        else:
            model_predict_int = int(model_predict_prob > 0.5)

        print(model_predict_prob)
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        else:
            print('file path does not exist')

        model_final_output = output_class_strings[model_predict_int]

        return render_template('model_result.html', message={'type': 'normal', 'text': f'Model predicts as <b>"{model_final_output}"</b> with probability {model_predict_prob}'})


@app.route('/sign_language_recognition')
def sign_language_recognition():
    model_summary = sign_language_model.to_json()
    # print(model_summary)
    return render_template("sign_language_detection.html", sign_model={"model_json": model_summary})


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def imageToArray(imageName):
    # Load the image and resize it to the desired dimensions
    # image_path = f'images/{imageName}'
    image_path = imageName
    width, height = 64, 64

    image = Image.open(image_path)

    if image.width < 64 or image.height < 64:
        return False

    image = image.resize((width, height))
    print(image.width)
    # Convert the image to a NumPy array and normalize the pixel values (if necessary)
    image_array = np.array(image)
    image_array = image_array / 255.  # Normalize the pixel values between 0 and 1

    # plt.imshow(image_array)
    # plt.show()

    # print(image_array.shape)
    # Reshape the image array to match the input shape of your model
    # Assumes the input shape is (width, height, 3)
    try:
        image_array = image_array.reshape(1, width, height, 3)
    except:
        if os.path.exists(image_path):
            os.remove(image_path)
        return False

    return image_array


@app.route('/black_friday')
def black_friday():
    return render_template('black_friday.html')


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
        print(input_data_final)

        return render_template('model_result.html', message={"type": 'normal', 'text': str(input_data_final)})


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
            return render_template('model_result.html', message={'type': 'error', 'text': f"Something went wrong!"})

        output_str = ["invalid", "valid"]

        return render_template('model_result.html', message={'type': 'normal', 'text': f"Model says it's a <b>{output_str[predict[0]]}</b> banknote."})


if __name__ == "__main__":
    # app.run(host='0.0.0.0')
    app.run(debug=True)
