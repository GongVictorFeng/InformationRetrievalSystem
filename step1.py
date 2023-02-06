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

def vetorSpaceModel(doc_dic):
    """this function is to create an inverted index, double dictionary is used, the key of vectorSpace is the token, the value of the vectorSpace
    is another dictionary, whose key is the document number, the value is the term frequency
    parameter: a dictionary of the document number and list of tokens
    return: a double dictionary, which is a vetor space model containing terms, documents(key of the inner dictionary) and df(length of the inner dictionary) and 
    tf(the value of the inner dictionary)"""
    vectorSpace={}
    for term in index:
        tfs={}
        dfNum=0
        for docNo in doc_dic:
            tf=doc_dic[docNo].count(term)
            if tf>0:
                tfs[docNo]=tf
                dfNum=dfNum+1
        vectorSpace[term]=tfs
    return vectorSpace





########### test #############

filePath="test.txt"
dic=createTerms(filePath)
vetorSpaceModel(dic)
