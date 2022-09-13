import sys
from sentence_transformers import SentenceTransformer, util


def compare(sentence, sentences_to_compare):
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Two lists of sentences
    sentences1 = [sentence]
    sentences2 = sentences_to_compare

    # Compute embedding for both lists
    embeddings1 = model.encode(sentences1, convert_to_tensor=True)
    embeddings2 = model.encode(sentences2, convert_to_tensor=True)

    # Compute cosine-similarities
    cosine_scores = util.cos_sim(embeddings1, embeddings2)

    # Output the pairs with their score
    # for i in range(len(sentences2)):
    #     print("{} \t\t {} \t\t Score: {:.4f}".format(sentences1[0], sentences2[i], cosine_scores[0][i]))
    # print(", ".join(str(round(x, 8)) for x in cosine_scores[0].tolist()))
    return cosine_scores.tolist()[0]
