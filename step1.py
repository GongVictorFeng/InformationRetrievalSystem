import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
############ variable ##############
index=set() # a set containing all terms, using set to remove duplicates

############ fuctions ###############

def createTerms(fileName):
    """This function is to extract the text from file and tokenize the text, then create a dictionary where the key is the DocNum, the value is the
        terms in the document; and a index containing terms of all documents.
        parameter: a file path
        return: a dictionary.
    """
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



########### test #############

filePath="test.txt"
dic=createTerms(filePath)
print(dic)
