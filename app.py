from flask import Flask, render_template

app = Flask(__name__)

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

    
]

@app.route("/")
def home():
    return render_template('home.html', mywork=my_work)

if __name__ == "__main__":
    #app.run(host='0.0.0.0')
    app.run(debug=True)