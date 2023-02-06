############### imports ###############
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

############### variables ###############
doc_text_pairs = {} # Dictionary of document name to document text pairs

############### functions ###############

# Function to separate file into documents
def separate_docs(f) :
    while True:
        line = f.readline()

        if not line:
            # Stop reading at end of file
            break
        elif (line.__contains__("<DOCNO>")) :
            # Isolate the title of the document
            doc_title = line[line.index("<DOCNO> ") + 8 : line.index(" </DOCNO>")]
        elif (line.__contains__("<TEXT>")) :
            doc_text = ""
            line = f.readline()
            while (not line.__contains__("</TEXT>")) :
                doc_text += line.replace("\n", " ")
                line = f.readline()
        elif (line.__contains__("</DOC>")) :
            # Turn plaintext into tokens (this reduces memory usage by ~20%)
            tokens = preprocessing(doc_text)
            # Save the key value pair
            doc_text_pairs[doc_title] = tokens
            # Reset values
            doc_title = ""
            doc_text = ""

# Function to handle processing of plaintext into vocabulary
def preprocessing(doc_text) :
    # Tokenization
    tokens = word_tokenize(doc_text)

    # Stopword Removal
    stop_words = set(stopwords.words("english"))
    filtered_list = []
    for word in tokens:
        if word.casefold() not in stop_words:
            filtered_list.append(word)

    # Remove numbers and punctuation
    filtered_list = [word.lower() for word in filtered_list if word.isalpha()]

    # Return results
    return filtered_list

############### main ###############

# Get a list of all files in the coll/ directory
lst_filenames = os.listdir("coll")

# Open each file one by one to separate the documents within the file
for filename in lst_filenames :
    file_path =  "coll\\" + filename
    f = open(file_path, "r")
    separate_docs(f)


# # For testing
# for a in doc_text_pairs :
#     print(a, doc_text_pairs[a])
#     print()