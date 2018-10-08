# -*- coding:utf-8 -*-
import numpy as np
import jieba
import re
import codecs
import time
import pandas as pd
from gensim.models import Word2Vec,KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
from nltk import word_tokenize

def LogInfo(stri):
    print(str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))+'  '+stri)
    
def preprocess_data_en(stopwords,doc):
    '''
    Function: preprocess data in Chinese including cleaning, tokenzing...
    Input: document string
    Output: list of words
    '''     
    doc = doc.lower()
    doc = word_tokenize(doc)
    doc = [word for word in doc if word not in set(stopwords)]
    doc = [word for word in doc if word.isalpha()]
    return doc

def preprocess_data_cn(stopwords,doc):
    '''
    Function: preprocess data in Chinese including cleaning, tokenzing...
    Input: 
        stopwords: Chinese stopwords list
        doc: document string
    Output: list of words
    '''       
    # clean data
    doc = re.sub(u"[^\u4E00-\u9FFF]", "", doc) # delete all non-chinese characters
    doc = re.sub(u"[儿]", "", doc) # delete 儿
    # tokenize and move stopwords 
    doc = [word for word in jieba.cut(doc) if word not in set(stopwords)]   
    return doc

def doc_vector(model,doc):
    '''
    Function:
        compute the mean of word vectors
    Input:
        model: gensim word2vec model
        doc: list of words
    Output:
        doc vector 
    '''
    # remove out-of-vocab words
    doc = [word for word in doc if word in model.vocab]
    return np.mean(model[doc],axis=0)

def has_representation(model,doc):
    '''
    Function:
        check if a document has a vector representation
    Input:
        doc: list of words
    Output:
        Boolean value:
            True: doc is not null and has vector representation
            False: doc is null or has no vector representation
    '''
    if len(doc)==0:
        # check if doc is null
        return False
    else:
        # check if at least one word of the document is in the word2vec dictionary
        return not all(word not in model.vocab for word in doc)
    
def calculate_similarity(model,doc1,doc2):
    '''
    Function:
        calculate cosine similarity of document pair
    Input: 
        model: gensim word2vec model
        doc1: list of words of document1
        doc2: list of words of document2
    Output:
        similarity of doc1 and doc2: (float)
            value ranges from 0 to 1;
            -1 means error
    '''
    # check representations 
    if not has_representation(model,doc1) or not has_representation(model,doc2):
        # if any of the two documents does not have representation
        return -1
    else:
        vec1 = np.array(doc_vector(model,doc1)).reshape(1,-1)
        vec2 = np.array(doc_vector(model,doc2)).reshape(1,-1)
        cos = cosine_similarity(vec1,vec2)[0][0]      
        # regularize value of cos to [-1,1]
        if cos<-1.0:cos=-1.0
        if cos>1.0:cos=1.0      
        sim = 1-np.arccos(cos)/np.pi 
        return sim

def regularize_sim(sims):
    '''
    Function: replace illegal similarity value -1 with mean value
    Input: list of similarity of document pairs
    Output: regularized list of similarity 
    '''
    sim_mean = np.mean([sim for sim in sims if sim!=-1])
    r_sims = []
    errors = 0
    for sim in sims:
        if sim==-1:
            r_sims.append(sim_mean)
            errors += 1
        else:
            r_sims.append(sim)
    LogInfo('Errors: '+str(errors))
    return r_sims
    
def doc_sim(lang,docs1,docs2):
    '''
    Function:
        calculate similarity of document pairs 
    Input: 
        lang: text language-Chinese for 'cn'/ English for 'en'
        docs1:  document strings list1
        docs2: document strings list2
    Output:
        similarity list of docs1 and docs2 pairs:
            value ranges from 0 to 1;
            -1 means error
    '''
    # check if the number of documents matched
    assert len(docs1)==len(docs2) ,'Documents number is not matched!'
    assert len(docs1)!=0,'Documents list1 is null'
    assert len(docs2)!=0,'Documents list2 is null'
    assert lang=='cn' or lang=='en', 'Language setting is wrong'
    # change setting according to text language 
    if lang=='cn':
        model_path = '../model/cn.cbow.bin'
        stopwords_path = '../data/chinese_stopwords.txt'
        preprocess_data = preprocess_data_cn
    elif lang=='en':
        model_path = '../model/GoogleNews-vectors-negative300.bin'
        stopwords_path = '../data/english_stopwords.txt'
        preprocess_data = preprocess_data_en
    # load word2vec model  
    LogInfo('Load word2vec model...')
    model = KeyedVectors.load_word2vec_format(model_path,binary=True,unicode_errors='ignore')
    # normalize vectors
    model.init_sims(replace=True)
    # preprocess data
    stopwords= [w.strip() for w in codecs.open(stopwords_path, 'r',encoding='utf-8').readlines()]
    sims = []
    LogInfo('Calculating similarity...')
    for i in range(len(docs1)):        
        p1 = preprocess_data(stopwords,docs1[i])
        p2 = preprocess_data(stopwords,docs2[i])
       # calculate similarity
        sim = calculate_similarity(model,p1,p2)
        sims.append(sim)
    # regularize sims
    r_sims = regularize_sim(sims)
    return r_sims

def main_cn():
    corpus = ['baidu_003_02','weixin_003_02','ifly_003_02',
              'baidu_008','weixin_008','ifly_008',
              'baidu_006_01','weixin_006_01', 'ifly_006_01',
              'baidu_004','weixin_004', 'ifly_004',
              'baidu_004_02','weixin_004_02','ifly_004_02',
               'baidu_rePunct_huiting','weixin_rePunct_huiting', 'ifly_rePunct_huiting']
#     corpus = ['170213']
    for c in corpus:
        LogInfo(c+' start')     
        # read data
        data = pd.read_csv('../data/'+c+'.csv')
        docs1 = data.REF.values
        docs2 = data.HYP.values
        # calculate similarity
        sims = doc_sim('cn',docs1,docs2)
        # save result as .xls
        save_path = '../res/'+c+'_w2v09.xls'
        res = pd.DataFrame(columns=['id','REF','HYP','semantic_similarity','SER','WER','difference'])
        res.id = data.id
        res.REF = docs1
        res.HYP = docs2
        res.WER = data.WER
        res.semantic_similarity = sims  
        res.SER = 1-res.semantic_similarity
        res.difference = res.SER-res.WER
        res.to_excel(save_path,index=0)
        LogInfo(c+' finish')

def main_en():
    LogInfo('Start')
    path1 = '../data/train_data1.txt'
    path2 = '../data/train_data2.txt'
    data1 = codecs.open(path1,'r',encoding='utf-8').read().split('\r\n')[:-1]
    data2 = codecs.open(path2,'r',encoding='utf-8').read().split('\r\n')[:-1]
    sims = doc_sim('en',data1,data2)
    # save result as .xls
    save_path = '../res/res_english_w2v.csv'
    res = pd.DataFrame(columns=['text1','text2','similarity'])
    res.text1 = data1
    res.text2 = data2
    res.similarity = sims
    res.to_csv(save_path,index=0)
    LogInfo('Save result as: '+save_path)

def example():
    # English text example
    docs1 = ['a speaker presents some products',
                 'vegetable is being sliced.',
                'man sitting using tool at a table in his home.']
    docs2 = ['the speaker is introducing the new products on a fair.',
                'someone is slicing a tomato with a knife on a cutting board.',
                'The president comes to China']
    sims = doc_sim('en',docs1,docs2)
    for i in range(len(sims)):
        print(docs1[i])
        print(docs2[i])
        print('Similarity: %.4f' %sims[i])
        
    # Chinese text example
    docs1 = ['做任何事都不能三天打鱼，两天晒网', 
             '学无止境', 
             '他整天愁眉苦脸']
    docs2 = ['对待每件事都不能轻言放弃', 
             '学海无涯，天道酬勤',
             '他和朋友去逛街']


    sims = doc_sim('cn',docs1,docs2)
    for i in range(len(sims)):
        print(docs1[i])
        print(docs2[i])
        print('Similarity: %.4f' %sims[i])
        
if __name__=='__main__':
#     main_cn()
    example()
