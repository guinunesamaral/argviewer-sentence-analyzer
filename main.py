from flask import request
from flask import Flask
from flask import jsonify
from sentence_similarity import compare
from profanities import check_profanity

app = Flask(__name__)


@app.route('/api/similarity', methods=["POST"])
def similarity():
    data = request.get_json()
    sentence = data["sentence"]
    sentences_to_compare = data["sentences_to_compare"]

    cosine_scores = compare(sentence, sentences_to_compare)
    return jsonify(cosine_scores)


@app.route('/api/profanity', methods=["POST"])
def profanity():
    data = request.get_json()
    sentence = data["sentence"]

    offense_score = check_profanity(sentence)

    return jsonify(offense_score)
