from flask import Flask, url_for
from flask import json
from flask import request
from prediction import prediction


app = Flask(__name__)

@app.route('/', methods = ['POST'])
def api_root():
    
    prediction(request.data)
    return 'Welcome'

@app.route('/articles/<articleid>')
def api_article(articleid):
    return 'You are reading ' + articleid

if __name__ == '__main__':
    app.run(host="localhost", port=8000, debug=True)
