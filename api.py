from flask import Flask, Response, json, request, url_for

from prediction import prediction

app = Flask(__name__)

@app.route('/predict', methods = ['POST'])
def api_root():
    datas = prediction(request.data)
    resp = Response(json.dumps(datas), status=200, mimetype='application/json')
    print (resp.status_code)
    return resp


if __name__ == '__main__':
    app.run(host="localhost", port=8000, debug=True)
