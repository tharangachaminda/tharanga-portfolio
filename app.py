from flask import Flask

app = Flask(__name__)

@app.route("/")
def home():
    return "<h1>Tharangas' Portfolio is running...</h1>"

if __name__ == "__main__":
    app.run(debug=True, use_reloader=True)