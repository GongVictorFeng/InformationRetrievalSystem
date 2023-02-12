import os
import math
import time #progress bar
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

def vectorSpaceModel(doc_dic):
    """this function is to create an inverted index, double dictionary is used, the key of vectorSpace is the token, the value of the vectorSpace
    is another dictionary, whose key is the document number, the value is the term frequency
    parameter: a dictionary of the document number and list of tokens
    return: a list containning 3 data, the first one is a double dictionary, which is a vector space model containing terms, documents(key of the inner dictionary) and df(length of the inner dictionary) and 
    tf(the value of the inner dictionary); the second one is a dictionary, the key is the token and the value is the idf of each token
    the last one is a list containing the vector length of each document"""
    
    vectorSpace={}
    idfDic={}
    N=len(doc_dic)
    maxTf=0 #max term frequency for normalization
    count=0 #progress bar
    num_total=len(index) #progress bar
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
        count+=1 #progress bar
        if (count%500==0):
            print("Creating Inverted Index:",count,"/",num_total,"terms") #progress bar
        

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
    infile.close()
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

    #retrieve all relevant docs using the inverted indexing
    relevantDocSet=set() #using set to remove duplicates
    for queryword in query:
        if queryword in vectorModel[0]:
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
    """This function is sorts the dictionary of documents by similarity value and stores the results as a list of tuples
    parameter: a dictionary of document number and similarity value
    return: an ordered list of tuples (document number, similarity value) sorted from greatest to least"""
    sortedDoc=sorted(similarity,reverse=True,key=similarity.get) #reverse sets to True means descending order
    rankedList=[]
    for doc in sortedDoc:
        rankedList.append((doc,similarity[doc]))
    return rankedList # TODO: return only top 1000 results for each query

def outputToFile(outfile, query_num,rankedList):
    """This function writes the results of a query into a file with the format specified by the assignment
    parameter: file to write into, number of the current query (1-50), list of documents ranked by similarity score
    returns: none"""
    for doc_rank in range(len(rankedList)):
        doc_num=str(rankedList[doc_rank][0])
        doc_score=str(rankedList[doc_rank][1])
        formattedQueryRank=str(query_num)+" Q0 "+" "+doc_num+" "+str(doc_rank+1)+" "+doc_score+" "+"tag_name"
        outfile.write(formattedQueryRank+"\n")


########### main #############

#get a list of all files in the coll/ directory
lst_filenames=os.listdir("coll")

#open each file one by one, separate the documents within file
#process into tokens and save (document number, tokens) as key-value pair
doc_dic={}
count=0 #progress bar
num_total=len(lst_filenames) #progress bar
startTime=time.perf_counter() #progress bar
for filename in lst_filenames:
    file_path="coll\\"+filename
    part_doc_dic=createTerms(file_path)
    doc_dic=doc_dic | part_doc_dic
    count+=1 #progress bar
    print("Preprocessing:",count,"/",num_total,"documents") #progress bar
endTime=time.perf_counter() #progress bar
print("Preprocessing elapsed time:",endTime-startTime,"seconds") #progress bar

#create inverted index from documents and tokens
startTime=time.perf_counter() #progress bar
vector=vectorSpaceModel(doc_dic)
endTime=time.perf_counter() #progress bar
print("Preprocessing elapsed time:",endTime-startTime,"seconds") #progress bar


#process queries into tokens
queryPath="topics1-50.txt"
queries=queryProcessor(queryPath)
print("Query vectors created") #progress bar

outfile=open("Results","w")
for query_num in range(len(queries)):
    similarity=retrieval(queries[query_num],vector) #retrieve relevant documents
    rankedList=ranking(similarity) #rank by similarity
    outputToFile(outfile,query_num,rankedList) #write results to file
    print("Query Number",query_num,"finished") #progress bar
outfile.close()

print("All done!") #progress bar

########### test #############

# filePath="test1.txt"
# doc_dic=createTerms(filePath)
# vector=vectorSpaceModel(doc_dic)
# queryPath="testQuery1.txt"
# queries=queryProcessor(queryPath)
# similarity=retrieval(queries[0],vector)
# ranking(similarity)