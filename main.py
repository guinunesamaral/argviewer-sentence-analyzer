from flask import request
from flask import Flask
from flask import jsonify

from sentence_similarity import compare

app = Flask(__name__)


@app.route('/api/similarity', methods=["POST"])
def index():
    data = request.get_json()
    sentence = data["sentence"]
    sentences_to_compare = data["sentences_to_compare"]

    cosine_scores = compare(sentence, sentences_to_compare)
    return jsonify(cosine_scores)
