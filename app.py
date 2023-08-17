from flask import Flask, render_template

app = Flask(__name__)

my_work = [
    {
        'title': 'SpaceX Falcon 9 1st Stage Landing Prediction - Analysis',
        'description': '''<ol>
                            <li>Python statistical libraries and Mathematical libraries for data analysis</li>
                            <li>SQL language</li>
                            <li>SQlite</li>
                            <li><a href="https://python-visualization.github.io/folium/">Folium</a> for interative Maps</li>
                            <li>Various Python libraries to visualize different insights along the way</li>
                            <li><a href="https://dash.plotly.com/">Plotly Dash</a> for interactive dashboard</li>
                            <li>Heroku deployment</li>
                          </ol>  
                            ''',
        'github': 'https://github.com/tharangachaminda/popularity-based-recommendation-system',
        'icons': ['Python', 'Jupyter-lab', 'SQL', 'SQLight', 'Folium', 'Plotly-dash', 'Heroku']
    },

    {
        'title': 'Black Friday Purchase Prediction',
        'description': '''<ol>
                            <li>Exploratory Data Analysis</li>
                            <li>Pandas</li>
                            <li>Flask</li>
                            <li>Heroku deployment</li>
                          </ol>  
                            ''',
        'github': 'https://github.com/tharangachaminda/Black_Friday_Purchase',
        'icons': ['Python', 'Jupyter-lab', 'SQL', 'SQLight', 'Folium', 'Plotly-dash', 'Heroku']
    },

    {
        'title': 'Netflix Recommender System (Content Based)',
        'description': '''<ol>
                            <li>NLP</li>
                            <li>NLTK Toolkit</li>
                            <li>Flask</li>
                            <li>Heroku deployment</li>
                          </ol>  
                            ''',
        'github': 'https://github.com/tharangachaminda/popularity-based-recommendation-system',
        'icons': ['Python', 'Jupyter-lab', 'SQL', 'SQLight', 'Folium', 'Plotly-dash', 'Heroku']
    },
]

@app.route("/")
def home():
    return render_template('home.html', mywork=my_work)

if __name__ == "__main__":
    #app.run(host='0.0.0.0')
    app.run(debug=True)