from flask import Flask, render_template, request, jsonify
from PIL import Image
import os
import numpy as np

import tensorflow as tf

app = Flask(__name__)

mood_class_strings = ['Sad', 'Happy']
model_obj = tf.keras.models.load_model('ML_models/happy_model.keras')
up_path = os.path.join(os.getcwd(), 'static', 'assets', 'temp')
print(up_path)

my_work = [
    {
        'header': 'End-to-end',
        'image': 'black_friday.png',
        'title': 'Black Friday Purchase Prediction',
        'description': "Banknote analysis refers to the examination of paper currency to determine its legitimacy and identify potential counterfeits. In this python project, I am trying to build a <b>Classification Machine Learning models</b> to predict banknotes are genuine or forged.",
        'github': 'https://github.com/tharangachaminda/banknotes_analysis',
        'icons': ['python', 'jupyterlab', 'flask', 'heroku']
    },

    {
        'header': 'End-to-end',
        'image': 'bank_notes.png',
        'title': 'Banknotes Authentication',
        'description': "Netflix Recommender system is one of the best recommender systems in the world. In this project I've used <b>movies datasets</b> and <b>Cosine similarity</b> to build this content based recommender system.",
        'github': 'https://github.com/tharangachaminda/content_based_recommender_system',
        'icons': ['python', 'jupyterlab', 'nltk', 'flask', 'heroku']
    },

    {
        'header': 'Analysis',
        'image': 'falcon_9.png',
        'title': 'SpaceX Falcon 9 1st Stage Landing Prediction',
        'description': "SpaceX re-uses the Stage 1 boosters of Falcon 9 rockets. This project is an analysis for successful Stage 1 landing prediction. I have used SpaceX API and Webscraping for data collection, SQlite database for data storage.",
        'github': 'https://github.com/tharangachaminda/popularity-based-recommendation-system',
        'icons': ['python', 'jupyterlab', 'sql', 'sqlite', 'folium', 'plotly-dash', 'heroku']
    },

    {
        'header': 'Recommender System',
        'image': 'netflix.png',
        'title': 'Netflix Recommender System (Popularity Based)',
        'description': "Netflix Recommender system is one of the best recommender systems in the world. In this project I've used <b>movies and rating</b> datasets and <b>NLTK toolkit</b> to build this popularity based recommender system.",
        'github': 'https://github.com/tharangachaminda/popularity-based-recommendation-system',
        'icons': ['Python', 'jupyterlab', 'flask', 'heroku']
    },

    {
        'header': 'Recommender System',
        'image': 'netflix.png',
        'title': 'Netflix Recommender System (Content Based)',
        'description': "Netflix Recommender system is one of the best recommender systems in the world. In this project I've used <b>movies datasets</b> and <b>Cosine similarity</b> to build this content based recommender system.",
        'github': 'https://github.com/tharangachaminda/content_based_recommender_system',
        'icons': ['Python', 'jupyterlabb', 'nltk', 'flask', 'heroku']
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
        'header': 'End-to-end',
        'image': 'mood_classifier.png',
        'title': 'Mood Classifier',
        'description': "A mood classification is a type of machine learning task that is used to recognize human moods or emotions. In this project I have implement a CNN model for recognizing smiling or not smiling humans using Tensorflow Keras Sequential API.",
        'github': 'https://github.com/tharangachaminda/cnn_mood_classifier',
        'icons': ['python', 'jupyterlab', 'tensorflow', 'flask', 'heroku']
    },

    {
        'header': 'End-to-end',
        'image': 'sign_language_digits.png',
        'title': 'Sign Language Digits Recognition',
        'description': "Sing language is a visual-gestural language used by deaf and hard-to-hearing individuals to convey imformation, thoughts and emotions. In this project I have implement a CNN model for recognizing sign language digits using Keras Functional API.",
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
    # print(model_summary)
    return render_template("mood_detection.html", mood_model={"model_obj": model_obj})


@app.route("/predict_mood", methods=["POST"])
def predict_mood():
    image = request.files['image_file']
    image.save(os.path.join(up_path, image.filename))
    imageArray = imageToArray(os.path.join(up_path, image.filename))

    mood_predict_prob = model_obj.predict(imageArray)
    mood_predict_int = int(mood_predict_prob > 0.5)

    print(mood_predict_prob)
    return mood_class_strings[mood_predict_int]


def imageToArray(imageName):
    # Load the image and resize it to the desired dimensions
    # image_path = f'images/{imageName}'
    image_path = imageName
    width, height = 64, 64

    image = Image.open(image_path)
    image = image.resize((width, height))
    # print(image.width)
    # Convert the image to a NumPy array and normalize the pixel values (if necessary)
    image_array = np.asarray(image)
    image_array = image_array / 255.  # Normalize the pixel values between 0 and 1

    # plt.imshow(image_array)
    # plt.show()

    # print(image_array.shape)
    # Reshape the image array to match the input shape of your model
    # Assumes the input shape is (width, height, 3)
    image_array = image_array.reshape(1, width, height, 3)

    return image_array


if __name__ == "__main__":
    # app.run(host='0.0.0.0')
    app.run(debug=True)
