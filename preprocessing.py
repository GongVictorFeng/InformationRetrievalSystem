# Implementing tokenization and stopword removal

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

sentence = "At eight o'clock on Thursday morning, Arthur didn't feel very good."

# Tokenization
tokens = word_tokenize(sentence)
print(tokens)

# Stopword Removal
stop_words = set(stopwords.words("english"))

filtered_list = []

for word in tokens:
    if word.casefold() not in stop_words:
        filtered_list.append(word)

print(filtered_list)

# Remove numbers and punctuation
filtered_list = [word.lower() for word in filtered_list if word.isalpha()]

print(filtered_list)