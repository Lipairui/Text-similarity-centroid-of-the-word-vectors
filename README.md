# Text similarity: centroid of the word vectors
Compute text similarity by calculating the cosine similarity of document vectors (Centroid of word vectors)

## Dependencies    
python 3.6.5   
numpy, gensim, sklearn, nltk, jieba, pandas, codecs, re

## Use pretrained word2vec model
Support both English and Chinese text format   
Chinese word2vec CBOW: utf8  2.18G   
http://pan.baidu.com/s/1qX334vE     
English word2vec 1.5G     
https://drive.google.com/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM&export=download

## Example usage
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

## Example results
### English:   
a speaker presents some products    
the speaker is introducing the new products on a fair.     
Similarity: 0.7321        

vegetable is being sliced.     
someone is slicing a tomato with a knife on a cutting board.     
Similarity: 0.6885    

man sitting using tool at a table in his home.     
The president comes to China     
Similarity: 0.5777  
### Chinese:      
做任何事都不能三天打鱼，两天晒网        
对待每件事都不能轻言放弃     
Similarity: 0.7043

学无止境      
学海无涯，天道酬勤       
Similarity: 0.6671        

他整天愁眉苦脸      
他和朋友去逛街      
Similarity: 0.5965
