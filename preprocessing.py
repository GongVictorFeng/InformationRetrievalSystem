############### imports ###############
import os
import time
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

############### variables ###############
doc_dic = {} # Dictionary of document name to document text pairs
index=set() # a set containing all terms, using set to remove duplicates

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
            doc_dic[doc_title] = tokens
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

    # Stemming w/ Porter Stemmer
    stemmer = PorterStemmer()
    filtered_list = [stemmer.stem(word) for word in filtered_list]

    # Return results
    index.update(tokens)
    return filtered_list

def createTerms(fileName):
    """This function is to extract the text from file and tokenize the text, then create a dictionary where the key is the DocNum, the value is the
        terms in the document; and a index containing terms of all documents.
        parameter: a file path
        return: a dictionary."""

    docs_dic={}
    infile=open(fileName)
    docStr=""
    isText=False
    for line in infile:
        if line.__contains__("<DOCNO> "):
            docNo=line[8:line.index(" </DOCNO>")]
        if line.__contains__("<TEXT>"):
            isText=True
            continue
        if line.__contains__("</TEXT>"):
            isText=False
        if isText:
            docStr+=line.replace("\n"," ")
        if line.__contains__("</DOC>"):
            lowcaseStr=docStr.casefold()
            tokens=preprocessing(lowcaseStr)
            index.update(tokens)
            docs_dic[docNo]=tokens
            docStr=""
    infile.close()
    return docs_dic

def preprocessing(str):
    """This function is to remove the no character word in the text sunch as number and punctuation, also remove the stopwords and stem each word
    parameter:a string to process
    return: a list containing tokens which have been processed"""

    tokenizer=RegexpTokenizer(r'[A-Za-z]+')
    tokens=tokenizer.tokenize(str)
    tokensWithoutStopwords=[token for token in tokens if token not in stopwords.words('english')]
    stemmer=PorterStemmer()
    stemmingTokens=[stemmer.stem(token) for token in tokensWithoutStopwords]
    return stemmingTokens

############### main ###############

# Get a list of all files in the coll/ directory
lst_filenames = os.listdir("coll")

#open each file one by one, separate the documents within file
#process into tokens and save (document number, tokens) as key-value pair
doc_dic={}
count = 0
startTime = time.perf_counter()
for filename in lst_filenames :
    file_path =  "coll\\" + filename
    f = open(file_path, "r")
    separate_docs(f)
    count+=1
    print("Preprocessing:", count, "/", len(lst_filenames))
endTime = time.perf_counter()
print("Preprocessing elapsed time:",endTime-startTime,"seconds") #progress bar

# doc_dic={}
# count=0 #progress bar
# num_total=len(lst_filenames) #progress bar
# startTime=time.perf_counter() #progress bar
# for filename in lst_filenames:
#     file_path="coll\\"+filename
#     part_doc_dic=createTerms(file_path)
#     doc_dic=doc_dic | part_doc_dic
#     count+=1 #progress bar
#     print("Preprocessing:",count,"/",num_total,"documents") #progress bar
# endTime=time.perf_counter() #progress bar
# print("Preprocessing elapsed time:",endTime-startTime,"seconds") #progress bar

# # For testing
# for a in doc_dic :
#     print(a, doc_dic[a])
#     print()