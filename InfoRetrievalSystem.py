import os
import math
import time #progress bar
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer, word_tokenize
############ variable ##############
index=set() # a set containing all terms, using set to remove duplicates

############ fuctions ###############

# Function to separate file into documents
def separate_docs(f):
    while True:
        line=f.readline()

        if not line:
            # Stop reading at end of file
            break
        elif (line.__contains__("<DOCNO>")):
            # Isolate the title of the document
            doc_title=line[line.index("<DOCNO> ")+8:line.index(" </DOCNO>")]
        elif (line.__contains__("<TEXT>")):
            doc_text=""
            line=f.readline()
            while (not line.__contains__("</TEXT>")):
                doc_text+=line.replace("\n", " ")
                line=f.readline()
        elif (line.__contains__("</DOC>")):
            # Turn plaintext into tokens (this reduces memory usage by ~20%)
            tokens=preprocessing(doc_text)
            # Save the key value pair
            doc_dic[doc_title]=tokens
            # Reset values
            doc_title=""
            doc_text=""

# Function to handle processing of plaintext into vocabulary
def preprocessing(doc_text):
    # Tokenization
    tokens=word_tokenize(doc_text)

    # Stopword Removal
    stop_words=set(stopwords.words("english"))
    filtered_list=[]
    for word in tokens:
        if word.casefold() not in stop_words:
            filtered_list.append(word)

    # Remove numbers and punctuation
    filtered_list=[word.lower() for word in filtered_list if word.isalpha()]

    # Stemming w/ Porter Stemmer
    stemmer=PorterStemmer()
    filtered_list=[stemmer.stem(word) for word in filtered_list]

    # Return results
    index.update(tokens)
    return filtered_list

def vectorSpaceModel(doc_dic):
    """this function is to create an inverted index, double dictionary is used, the key of vectorSpace is the token, the value of the vectorSpace
    is another dictionary, whose key is the document number, the value is the term frequency
    parameter: a dictionary of the document number and list of tokens
    return: a list containning 3 data, the first one is a double dictionary, which is a vector space model containing terms, documents(key of the inner dictionary) and df(length of the inner dictionary) and 
    tf(the value of the inner dictionary); the second one is a dictionary, the key is the token and the value is the idf of each token
    the last one is a list containing the vector length of each document"""
    
    #count frequency of each word in its respective document, stored double dictionary
    #format: {document_num : {token : token_freq}}
    doc_token_freqs={}
    for docNo in doc_dic:
        doc_token_freqs[docNo]={}
        for token in doc_dic[docNo]:
            if (token in doc_token_freqs[docNo]):
                doc_token_freqs[docNo][token]+=1
            else:
                doc_token_freqs[docNo][token]=1

    #build vector space
    vectorSpace={}
    idfDic={}
    num_documents=len(doc_dic)
    maxTf=0 #max term frequency for normalization
    for docNo in doc_token_freqs:
        for token in doc_token_freqs[docNo]:
            tf=doc_token_freqs[docNo][token]
            maxTf=max(maxTf,tf) #saving max tf for normalization
            if (token not in vectorSpace):
                vectorSpace[token]={}
            vectorSpace[token][docNo]=tf

    #calculate each term's idf
    for token in vectorSpace:
        dfNum=len(vectorSpace[token]) #len(dict) has O(1) according to Python Libraries
        idfDic[token]=math.log(num_documents/dfNum,2)

    #calculate document vector length
    vectorLength={}
    for docNo in doc_token_freqs:
        vectorLen=0
        for token in doc_token_freqs[docNo]:
            tf=doc_token_freqs[docNo][token]
            weight=(tf/maxTf)*idfDic[token]
            vectorLen+=weight**2
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
        if (queryTerm in vectorModel[1]):
            weight=(queryTermWeights[queryTerm]/maxTf)*vectorModel[1][queryTerm]
        else:
            weight=0
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
    for i in range(min(1000,len(sortedDoc))):
        doc=sortedDoc[i]
        rankedList.append((doc,similarity[doc]))
    return rankedList

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
    f.close()
endTime = time.perf_counter()
print("Preprocessing elapsed time:",endTime-startTime,"seconds") #progress bar

#create inverted index from documents and tokens
startTime=time.perf_counter() #progress bar
vector=vectorSpaceModel(doc_dic)
endTime=time.perf_counter() #progress bar
print("Indexing elapsed time:",endTime-startTime,"seconds") #progress bar

#process queries into tokens
startTime=time.perf_counter() #progress bar
queryPath="topics1-50.txt"
queries=queryProcessor(queryPath)
print("Query vectors created") #progress bar

outfile=open("Results","w")
for query_num in range(len(queries)):
    similarity=retrieval(queries[query_num],vector) #retrieve relevant documents
    rankedList=ranking(similarity) #rank by similarity
    outputToFile(outfile,query_num,rankedList) #write results to file
outfile.close()

endTime=time.perf_counter() #progress bar
print("Query processing, ranking, retrieval elapsed time:",endTime-startTime,"seconds") #progress bar

print("All done!") #progress bar

########### test #############

# filePath="test1.txt"
# doc_dic=createTerms(filePath)
# vector=vectorSpaceModel(doc_dic)
# queryPath="testQuery1.txt"
# queries=queryProcessor(queryPath)
# similarity=retrieval(queries[0],vector)
# ranking(similarity)