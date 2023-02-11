import nltk
import math
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
    return: a list containning 3 data, the fist one is a double dictionary, which is a vetor space model containing terms, documents(key of the inner dictionary) and df(length of the inner dictionary) and 
    tf(the value of the inner dictionary); the second one is a dictionary, the key is the token and the value is the idf of each token
    the last one is a list containing the vector length of each document"""
    
    vectorSpace={}
    idfDic={}
    N=len(doc_dic)
    maxTf=0 #max term frequency for normalization
    for term in index:
        tfs={}
        dfNum=0
        for docNo in doc_dic:
            tf=doc_dic[docNo].count(term)
            if tf>maxTf:
                maxTf=tf
            if tf>0:
                tfs[docNo]=tf
                dfNum=dfNum+1  
        #calculate idf of each term   
        idfDic[term]=math.log(N/dfNum,2)
        vectorSpace[term]=tfs

    #calculate vector length
    vectorLength={}
    for docNo in doc_dic:
        vectorLen=0
        for token in vectorSpace:
            try:
              weight=(vectorSpace[token][docNo]/maxTf)*idfDic[token] 
              vectorLen+=weight**2
            except:
                continue
        vectorLen=math.sqrt(vectorLen)
        vectorLength[docNo]=vectorLen

    return [vectorSpace,idfDic,vectorLength]

def queryProcessor(fileName):
    """This function is to extract the text from query file and tokenize the text, then create a 2D list where the row is query, the column is the
    keyword in the query.
    parameter: a file path
    return: a 2D list."""

    infile=open(fileName)
    queries=[] # a 2d list, the row is the queries, the column is the keywords each query contains.
    docStr=""
    isText=False
    for line in infile:
        if line.__contains__("<title>"):
            docStr+=line[6:]
        if line.__contains__("<desc>") or line.__contains__("<narr>"):
            isText=True
            continue
        if line.__contains__("</top>"):
            isText=False
            lowcaseStr=docStr.casefold()
            tokens=preprocessing(lowcaseStr)
            queries.append(tokens)
            docStr=""
        if isText:
            docStr+=line.replace("\n"," ")
    return queries

def createQueryVector(query,vectorModel):
    """This function is to compute the tf-idf weight for the query and the query length
    parameter: a list containing keywords of a query and the vector space model which contains the idf that can be used to compute weight
    return: a list, the first element is a dictionary, the key is the keyword, the value is the tf-idf weight; the second element is the query length"""

    #Calculate the tf-idf of each term in the query and the length of the query
    queryLength=0
    queryTermWeights={}
    maxTf=0 #max term frequency for normalization
    #calculate tf
    for term in index:
        tf=query.count(term)
        if tf>maxTf:
            maxTf=tf
        if tf>0:
            queryTermWeights[term]=tf
    #calculate tf-idf and query length
    for queryTerm in queryTermWeights:
        #vectorModel[1] is the dictionary of idf,the key is the term in the index
        weight=(queryTermWeights[queryTerm]/maxTf)*vectorModel[1][queryTerm]
        queryTermWeights[queryTerm]=weight
        queryLength+=weight**2
    queryLength=math.sqrt(queryLength)
    return [queryTermWeights,queryLength]

def retrieval(query,vectorModel):
    """This function is to retrieve all relevant data and calculate the similarity of each document
    parameter: the first one is the keyword list of the query, the second one is the vector space model of the collection
    return: a dictionary, the key is the document identity, the value is the similarity"""

    #get query vector and the query length
    queryVector=createQueryVector(query,vectorModel)

    #retrieve all relevant doc using the invert indexing
    relevantDocSet=set() #using set to remove duplicate
    for queryword in query:
        relevantDocs=vectorModel[0][queryword]
        for doc in relevantDocs:
            relevantDocSet.add(doc) #add relevant doc to the set
    
    #Calculate similarity for each relevant documents
    similarities={}
    for doc in relevantDocSet:
        similarity=0
        for word in queryVector[0]:
            #multiply the corresponding item in query vector and document vector, and then add all corresponding values to get the similarity of each doc
            try:
                similarity+=vectorModel[0][word][doc]*vectorModel[1][word]*queryVector[0][word]
            except:
                continue
        similarities[doc]=similarity/(vectorModel[2][doc]*queryVector[1])
    return similarities

def ranking(similarity):
    """this function is simply sort the dictionary of documents according to the similarity value """
    sortedDoc=sorted(similarity,reverse=False) #reverse sets to False means descending order
    print(sortedDoc)

########### test #############

filePath="test1.txt"
doc_dic=createTerms(filePath)
vector=vetorSpaceModel(doc_dic)
queryPath="testQuery1.txt"
queries=queryProcessor(queryPath)
similarity=retrieval(queries[0],vector)
ranking(similarity)
