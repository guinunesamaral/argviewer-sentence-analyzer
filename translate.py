from profanity_check import predict, predict_prob
import translators as ts

str = 'Negros devem ser exterminadas'

str = ts.google(str, from_language="auto", to_language="en")

print(str)

is_offensive = bool(predict([str])[0])

print(is_offensive)
print(predict_prob([str])[0])
