from profanity_check import predict, predict_prob
import translators as ts


def check_profanity(str):
    str = ts.google(str, from_language="auto", to_language="en")
    return predict_prob([str])[0]
