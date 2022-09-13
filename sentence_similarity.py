from sentence_transformers import SentenceTransformer, util


def round_first_two_decimals(score): return round(score, 2)


def compare(sentence, sentences_to_compare):
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Compute embedding for both lists
    embeddings1 = model.encode([sentence], convert_to_tensor=True)
    embeddings2 = model.encode(sentences_to_compare, convert_to_tensor=True)

    # Compute cosine-similarities
    cosine_scores = util.cos_sim(embeddings1, embeddings2)

    return list(map(round_first_two_decimals, cosine_scores.tolist()[0]))
