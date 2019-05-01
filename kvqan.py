#!/usr/bin/env python
# coding: utf-8

# In[2]:


#four modules:

# 1.Question module
# 2. Answer module
# 3. Fact module
# 4. memory module


# tasks to do:
# 1. get the 4 most relevant facts for each question.
# 2. remove question mark from the 

# max question length-->91, maximum fact length-->99, number of facts -->4 , amaximum ans len==11


# In[1]:


#---------------------------------importing libaries ------------------------------------------#
import numpy as np
import json
import pandas as pd
import ast
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim


# In[2087]:


#df.to_csv('data_to_use.csv')


# In[2088]:


'''
data_to_use=pd.read_csv('data_to_use.csv')
df=data_to_use
'''


# # First part--> loading data, Getting relevant facts

# In[3]:


#------------------------------------Loading dataset --------------------------------------#
def load_dataset():
    with open('datasets/dataset.json') as json_file:  
        data = json.load(json_file)
    #print(type(data['21717']))
    #print(type(data))
    labels=[]
    for line in data['21717']:
        labels.append(line)
    ids=[]
    for ide in data.keys():
        ids.append(ide)
    return data


# In[4]:


def load_kb():
    #-------------------------Loading Knowledge base ---------------------------#
    fa=pd.read_csv("datasets/KGfacts/a1.csv",sep='\t', error_bad_lines=False)
    fa=fa.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
    facts=fa
    
    #----------------Loading mapping between Qids and entities -----------------#
    df = pd.read_csv('datasets/KGfacts/Qid-NamedEntityMapping.csv',sep='\t',error_bad_lines=False, encoding='ascii')
    df.columns=["id","name"]
    names=df["name"].apply(ast.literal_eval).apply(bytes.decode)
    df["name"]=list(names)
    mapping=df
    facts=facts.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
    mapping=mapping.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
    return facts,mapping


# In[5]:


def id_with_name(ide):
    
    name_=mapping[mapping['id']==ide]
    x=pd.DataFrame(name_["name"])
    s_find=str(x['name']).split("\n")[0]
    s_find = re.sub(r'^[\d-]+ ', '', s_find, 1)
    s_find=s_find[3:]
    return s_find


# In[6]:


def get_facts(question,NamedEntity):
    fc=[]
    for entity in NamedEntity:
        qid=mapping[mapping['name']==entity]
        ide=qid["id"]
        ide=pd.DataFrame(ide)
    #print(ide)
        s_find=str(ide['id']).split("\n")[0]
        index=s_find.find("Q")
        s_find=s_find[index:]
    #print(s_find)
        rel_facts=facts[facts['entity']==s_find]
    #print(rel_facts)
        rel_facts_list=[]
        for index,fact in rel_facts.iterrows():
            rel_facts_list.append([fact["entity"],fact["relation"],fact["attribute"]])    
    #print(rel_facts_list)
        rel=[]
        for fact in rel_facts_list:
            #print(fact)
            rel.append(" ".join(fact))
    #print(rel)
        rel1=[]
        for fact in rel:
            rel1.append(fact.split(" "))
        fc+=rel1
    return(fc)
    


# In[7]:


def get_emb_facts(rel_facts):
    embedded_facts=[]
    for fact in rel_facts:
        em=[]
        for word in fact:  
            if word in embeddings_index:    
                em.append( embeddings_index[word])     
        embedded_facts.append(em)
    return embedded_facts


# In[8]:


def  get_final_avg(embedded_facts):
    embedded_facts_arr=[]
    for fact in embedded_facts:
        embedded_facts_arr.append(np.array(fact))
    embedded_facts_arr=np.array(embedded_facts_arr)
    embedded_facts_arr_avg=[]
    for fact in embedded_facts_arr:
        embedded_facts_arr_avg.append(np.mean(fact,axis=0))
    embedded_facts_arr_avg=np.array(embedded_facts_arr_avg)
    return embedded_facts_arr_avg


# In[9]:


def load_embedding_matrix():
    embeddings_index = {}
    f = open('glove.6B.100d.txt')
    for line in f:
        values = line.split(' ')
        word = values[0] ## The first entry is the word
        coefs = np.asarray(values[1:], dtype='float32') ## These are the vecotrs representing the embedding for the word
        embeddings_index[word] = coefs
    f.close()
    print('GloVe data loaded') 
    return embeddings_index
    
    


# In[10]:


def get_embedding(word):
    return embeddings_index[word]


# In[11]:


#------------------------------Loading data --------------------------------------#


data=load_dataset()
#------------------------------Loading knowledge base and mapping --------------------------------------#

facts,mapping=load_kb()
#------------------------------Loading embedding matrix --------------------------------------#

embeddings_index=load_embedding_matrix()


#------------------------------one time deal--------------------------------------#


# In[12]:




processed_facts_entity=[]
for index,row in facts.iterrows():
    inner=[]
    for word in (row['entity'].split(" ")):
        inner.append((word).replace('Ñ','').replace('¡','').replace('å',' ').replace('°',' ').replace('Ð','').replace('?','').replace(':',' ').replace(',',' ').replace('[',' ').replace('{',' ').replace(']','').replace('}',' ').replace('"','').replace('\\',"").replace("'","").replace('.',''))
        
    processed_facts_entity.append(inner)  
        

        
processed_facts_relation=[]
for index,row in facts.iterrows():
    inner=[]
    for word in (row['relation'].split(" ")):
        inner.append((word).replace('Ñ','').replace('¡','').replace('å',' ').replace('°',' ').replace('Ð','').replace('?','').replace(':',' ').replace(',',' ').replace('[',' ').replace('{',' ').replace(']','').replace('}',' ').replace('"','').replace('\\',"").replace("'","").replace('.',''))
        
    processed_facts_relation.append(inner)  
    
    
    
processed_facts_attribute=[]
for index,row in facts.iterrows():
    inner=[]
    for word in (row['attribute'].split(" ")):
        inner.append((word).replace('\n',' ').replace('Ñ','').replace('¡','').replace('å',' ').replace('°',' ').replace('Ð','').replace('?','').replace(':',' ').replace(',',' ').replace('[',' ').replace('{',' ').replace(']','').replace('}',' ').replace('"','').replace('\\',"").replace("'","").replace('.',''))
        
    processed_facts_attribute.append(inner) 
    
fr=[]
for word in processed_facts_attribute:
    for w in word:
        fr.append(w)
for word in processed_facts_relation:
    for w in word:
        fr.append(w)
for word in processed_facts_entity:
    for w in word:
        fr.append(w)
        
fr_=[]
for word in fr:
    for w in word.split(" "):
        fr_.append(w)



total_facts=[]
for i in range(len(processed_facts_entity)):
    total_facts.append([" ".join(str(x) for x in processed_facts_entity[i]),
                        " ".join(str(x) for x in processed_facts_relation[i]),
                                 " ".join(str(x) for x in processed_facts_attribute[i])])
    
    
    
total_facts=pd.DataFrame(total_facts,columns=['entity','relation','attribute']) 



    


# In[25]:


for fact in processed_facts_attribute:
    for word in fact:
        if("." in word):
            print(fact)


# In[13]:


facts=total_facts


# In[14]:


# dats is stored in form of dictionary of dictionaries.
#outer dictionary is index: {data[index]}
#inner dictionary is named_entities:[---,---,---],questions:[---,---,],....
# we have to get index for each word
#------------------------------getting inidices of data dictionary -----------------------------------#

indices=list(data.keys())
indices.sort()
train_indices=indices[:20000]
test_indices=indices[20000:]


# In[15]:


#------------------------------class for indexing all words -----------------------------------#

class vocab:
    def __init__(self):
        self.n_words=2
        self.word2index={}
        self.word2count={}
        self.index2word={}
        self.index2word = {0: "<EOS>", 1: "<PAD>"}
    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word]=self.n_words
            self.word2count[word]=1
            self.index2word[self.n_words]=word
            self.n_words=self.n_words+1
        else:
            self.word2count[word]+=1
        
    def index_words(self, sentence):
        for word in sentence:
            self.index_word(word)


# In[16]:


data_class=vocab()


# In[17]:


def index_from_sentence(sentence,type_):
    indexes=[word_dict_usable[word] for word in sentence.split(" ") or sentence.split("\n") ]
    if(type_=="Q"):
        length=92
        
        indexes+=[0 for i in range(len(indexes),length)]
        indexes[length-1]=1
    elif(type_=="A"):
        length=12
        indexes+=[0 for i in range(len(indexes),length)]
        indexes[length-1]=1
    elif(type_=="F"):
        length=120
        indexes+=[0 for i in range(len(indexes),length)]
        indexes[length-1]=1
    elif(type_=="Ne"):
        length=92
        indexes+=[0 for i in range(len(indexes),length)]
        indexes[length-1]=1
    else:
        print("dear lord")
    return indexes

def variable_from_sentence(sentence,type_):
    index=index_from_sentence(sentence,type_)  
    var = Variable(torch.LongTensor(index))
    return var


# # Question Module

# In[26]:


class QuestionModule(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(QuestionModule, self).__init__()
        self.gru=nn.GRU(hidden_size,hidden_size,bidirectional=False,batch_first=True)

    def forward(self, questions, word_embedding):
        questions = word_embedding(questions)
        _, questions = self.gru(questions)
        questions = questions.transpose(0, 1)
        return questions


# In[27]:


def get_embed_facts(question, NamedEntity):
    fv=get_facts(question,NamedEntity)
    embedded_facts=get_emb_facts(fv)
    emded_mean=get_final_avg(embedded_facts)
    return fv,torch.as_tensor(emded_mean)


# In[28]:


def get_fact_index(question_rep,final_facts):
    sim=[]
    a=question_rep
    a = torch.tanh(question_rep)
    a = torch.tensor(a.tolist())
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    for index,fact in enumerate(final_facts):
        b = torch.tanh(fact)
        b = torch.tensor([b.tolist()])
        sim+=(cos(a,b).tolist())
    fact_index=sorted(range(len(sim)), key=lambda i: sim[i])[-4:]
    return fact_index


# In[29]:


def get_4_facts(final_facts,fact_index):
    rel_facts=[]
    fact_index.sort()
    for index in fact_index:
        rel_facts.append(final_facts[index])
    return rel_facts
      


# In[30]:


def getFacts(question,NamedEntity):
    var=variable_from_sentence(question,type_="Q")
    var = Variable(var.unsqueeze(dim=0))
    question_module=QuestionModule(vocab_size,hidden_size)
    word_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0, sparse=True)
    question_rep=question_module(var,word_embedding)
    fv,final_facts=get_embed_facts(question, NamedEntity)
    fact_index=get_fact_index(question_rep,final_facts)
    embeded_most_rel_facts_nl=get_4_facts(fv,fact_index)
    embeded_most_rel_facts_emb=get_4_facts(final_facts,fact_index)
    return embeded_most_rel_facts_nl


# In[31]:


labels=['NamedEntities','Answers','Questions','wikiCap']


# # Most important lines of code

# # Preparing data for training KVQAN

# In[59]:



new_data=dict()
for index in indices:
    inside_dic=dict()
    new_data[index]=inside_dic
    for label in labels:
        label_list=[]
        size=(len(data[index][label]))   
        for word in data[index][label]:
            #label_list.append(re.sub(r'[^A-Za-z0-9 ]', '', str(word))) 
                label_list.append(repr(str(word)).replace('Ñ','').replace('¡','')
                .replace('°',' ').replace('Ð','').replace('?','').replace(':',' ').replace(',',' ')
                                 .replace('[',' ').replace('{',' ').replace(']','').replace('}',' ')
                                  .replace('"','').replace('\\',"").replace("'","").replace(".",""))
                       
        inside_dic[label]=label_list
        
import re
         


# In[60]:


#-----------------function for getting the data in right format so we can use above class ------------------#
def index_data():
    
    pre_data=[]
    for index in indices:
        for word in str(new_data[index]).split(" "):
            pre_data.append((word))
    #print(pre_data[0:100])
    clean_data=[]
    for word in pre_data:
        clean_data.append(word.replace('?',' ').replace(':',' ').replace(',',' ').replace('[',' ').replace('{',' ').replace(']',' ').replace('}',' ').replace('"','').replace("\\","").replace("'",""))
    #print((clean_data[0:200]))
    data_to_index=clean_data+fr_
    data_class.index_words(data_to_index)
    
    #data_class.index_words(fact_words)
    return data_class
    
        
        

process_data_class=index_data()


# In[61]:


word_dict=dict()
for value in (process_data_class.index2word):
    word_dict[value]=(process_data_class.index2word[value]).replace(" ","")

word_dict_usable = {v: k for k, v in word_dict.items()}
word_dict_usable


# In[64]:


q_f_a=[]
for index in train_indices:
    for i in range(len(new_data[index]['Questions'])):
        q_f_a.append([new_data[index]['Questions'][i],new_data[index]['NamedEntities'],new_data[index]['Answers'][i]])
qfa=q_f_a
df = pd.DataFrame(qfa)
df.columns = ['question','NamedEntities' ,'answer']


# In[65]:


vocab_size=process_data_class.n_words
hidden_size=100


# In[ ]:



add_col=[]
for index, row in df.iterrows():
    print(index)
    add_col+=([getFacts(row['question'], row['NamedEntities'])])
 


# In[ ]:


import pickle

with open('outfile', 'wb') as fp:
    pickle.dump(add_col, fp)


with open ('outfile', 'rb') as fp:
    itemlist = pickle.load(fp)


# In[91]:



replaced_names=[]
for i in range(len(itemlist)):
    print(i)
    inner_replaced_names=[]
    for j in range(len(itemlist[i])):
        inner_replaced_names.append([id_with_name(itemlist[i][j][0])]+itemlist[i][j][1:])
    replaced_names.append(inner_replaced_names)


# In[93]:


replaced_names_str=[]
for name in replaced_names:
    inner_replaced_names_str=[]
    for row in name:
        inner_replaced_names_str.append(" ".join(str(x) for x in row))
    replaced_names_str.append(inner_replaced_names_str)
replaced_names_str


# In[34]:



df['related_str']=replaced_names_str


# In[120]:


replaced_names_str


# In[94]:


fact_var=[]

for facts in replaced_names_str:
    inner=[]
    for fact in facts:
        #for word in fact.split(" "):
            #if(word.find('\\xa0')!= -1):
             #   fact1 = fact.strip(word)
                
            
        inner.append(variable_from_sentence(fact,type_="F"))
        
        
    fact_var.append(inner)


# In[1575]:


criterion=nn.NLLLoss()
max_length=12
answer_rnn=Answer_Module_RNN(hidden_size,output_size)
input_variable=question
optimizer = optim.Adam(answer_rnn.parameters(), lr=0.001)
target_variable=answer


# In[1652]:


class Answer_Module_RNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, dropout_p=0.1):
        super(Answer_Module_RNN, self).__init__()
        
       
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        
        # Define layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size , hidden_size, n_layers, dropout=dropout_p)
        self.out = nn.Linear(hidden_size , output_size)
        
    
    def forward(self, word_input,fact_embedding,question_embedding
               ):
        word_embedded = self.embedding(word_input).view(1, 1, -1) 
        
        # Combine embedded input word and last context, run through RNN
        rnn_input=word_embedded
        rnn_output, hidden = self.gru(rnn_input, fact_embedding)
        rnn_output = rnn_output.squeeze(0) 
        output = F.log_softmax(self.out(rnn_output))
        
        return output,hidden


# In[1653]:


#-----------------getting word from index ------------------#

def get_sent_from_variable(var):
    sent=[]
    if(len(var)>1):
        for i in (var):
            i=i.tolist()
            sent.append(process_data_class.index2word[i[0]])
    else:
        var=var.tolist()
        sent.append(process_data_class.index2word[var[0]])
    return sent  


# In[1654]:


#-------------------train function --------------------#

def train(input_variable, target_variable,answer_rnn , optimizer, criterion, max_length=12):

    optimizer.zero_grad()
    loss = 0 
    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]
    question_module = QuestionModule(vocab_size, hidden_size)
    word_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0, sparse=True)
    Question=input_variable
    rnn_hidden=question_module(Question,word_embedding)   
        # Teacher forcing: Use the ground-truth target as the next input
    for di in range(target_length):
        rnn_input = target_variable[di]  
        rnn_output, rnn_hidden = answer_rnn(rnn_input,rnn_hidden)
        y = torch.argmax(rnn_output, dim=1)
        loss += criterion(rnn_output, target_variable[di].unsqueeze(dim=0))

    # Backpropagation    
    loss.backward()
    optimizer.step() 
    return loss.data/ target_length


# In[1655]:


#train(input_variable, target_variable,answer_rnn , optimizer, criterion, max_length=12)


# In[1674]:


n_epochs=30
plot_every = 1
print_every = 1
import time
import math
# Keep track of time elapsed and running averages
start = time.time()
plot_losses = []
print_loss_total = 0 # Reset every print_every
plot_loss_total = 0 # Reset every plot_every


# In[1675]:


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))


# In[ ]:


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')

def show_plot(points):
    plt.figure()
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(base=0.2) # put ticks at regular intervals
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

show_plot(plot_losses)


# In[1676]:




for epoch in range(1, n_epochs + 1):
    for index,data in enumerate(Data):
        facts,Question,Answer=data
        facts = Variable(facts.unsqueeze(dim=0))
        Question = Variable(Question.unsqueeze(dim=0))
        Answer = Variable(Answer)
        loss=train(Question, Answer,answer_rnn , optimizer, criterion, max_length=12)
        print_loss_total += loss
        plot_loss_total += loss

        if epoch == 0: continue

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print_summary = '%s (%d %d%%) %.4f' % (time_since(start, epoch / n_epochs), epoch, epoch / n_epochs * 100, print_loss_avg)
            print(print_summary)

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
            
            


# In[1004]:


vocab_size=process_data_class.n_words


# In[1180]:


emp_list=[]
for index,row in df.iterrows():
    emp_list.append(variable_from_sentence(row['answer'],type_="A"))
A=emp_list

temp_list=[]
for index,row in df.iterrows():
    temp_list.append(variable_from_sentence(row['question'],type_="Q"))
Q=temp_list


# In[1201]:


Data=[]
for i in range(len(A)):  
    Data.append([Q[i],Q[i],A[i]])


# In[ ]:


'''
things left:
1. facts embedding.
3. think of how you will incorporate these two things
4. attention
5. save model
6. predict for test set and calculate top-n metric and report
I 
'''


# In[1968]:



class AttentionGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AttentionGRUCell, self).__init__()
        self.hidden_size = hidden_size
        self.Wr = nn.Linear(input_size, hidden_size)
        init.xavier_normal(self.Wr.state_dict()['weight'])
        self.Ur = nn.Linear(hidden_size, hidden_size)
        init.xavier_normal(self.Ur.state_dict()['weight'])
        self.W = nn.Linear(input_size, hidden_size)
        init.xavier_normal(self.W.state_dict()['weight'])
        self.U = nn.Linear(hidden_size, hidden_size)
        init.xavier_normal(self.U.state_dict()['weight'])

    def forward(self, fact, C, g):
        '''
        fact.size() -> (#batch, #hidden = #embedding)
        c.size() -> (#hidden, ) -> (#batch, #hidden = #embedding)
        r.size() -> (#batch, #hidden = #embedding)
        h_tilda.size() -> (#batch, #hidden = #embedding)
        g.size() -> (#batch, )
        '''

        r = F.sigmoid(self.Wr(fact) + self.Ur(C))
        h_tilda = F.tanh(self.W(fact) + r * self.U(C))
        g = g.unsqueeze(1).expand_as(h_tilda)
        h = g * h_tilda + (1 - g) * C
        return h


# In[1969]:



class AttentionGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AttentionGRU, self).__init__()
        self.hidden_size = hidden_size
        self.AGRUCell = AttentionGRUCell(input_size, hidden_size)

    def forward(self, facts, G):
        '''
        facts.size() -> (#batch, #sentence, #hidden = #embedding)
        fact.size() -> (#batch, #hidden = #embedding)
        G.size() -> (#batch, #sentence)
        g.size() -> (#batch, )
        C.size() -> (#batch, #hidden)
        '''
        batch_num, sen_num, embedding_size = facts.size()
        C = Variable(torch.zeros(self.hidden_size))
        for sid in range(sen_num):
            fact = facts[:, sid, :]
            g = G[:, sid]
            if sid == 0:
                C = C.unsqueeze(0).expand_as(fact)
            C = self.AGRUCell(fact, C, g)
        return C


# In[1970]:





class EpisodicMemory(nn.Module):
    def __init__(self, hidden_size):
        super(EpisodicMemory, self).__init__()
        self.AGRU = AttentionGRU(hidden_size, hidden_size)
        self.z1 = nn.Linear(4 * hidden_size, hidden_size)
        self.z2 = nn.Linear(hidden_size, 1)
        self.next_mem = nn.Linear(3 * hidden_size, hidden_size)
        init.xavier_normal(self.z1.state_dict()['weight'])
        init.xavier_normal(self.z2.state_dict()['weight'])
        init.xavier_normal(self.next_mem.state_dict()['weight'])

    def make_interaction(self, facts, questions, prevM):
        '''
        facts.size() -> (#batch, #sentence, #hidden = #embedding)
        questions.size() -> (#batch, 1, #hidden)
        prevM.size() -> (#batch, #sentence = 1, #hidden = #embedding)
        z.size() -> (#batch, #sentence, 4 x #embedding)
        G.size() -> (#batch, #sentence)
        '''
        batch_num, sen_num, embedding_size = facts.size()
        questions = questions.expand_as(facts)
        prevM = prevM.expand_as(facts)

        z = torch.cat([
            facts * questions,
            facts * prevM,
            torch.abs(facts - questions),
            torch.abs(facts - prevM)
        ], dim=2)

        z = z.view(-1, 4 * embedding_size)

        G = F.tanh(self.z1(z))
        G = self.z2(G)
        G = G.view(batch_num, -1)
        G = F.softmax(G)

        return G

    def forward(self, facts, questions, prevM):
        '''
        facts.size() -> (#batch, #sentence, #hidden = #embedding)
        questions.size() -> (#batch, #sentence = 1, #hidden)
        prevM.size() -> (#batch, #sentence = 1, #hidden = #embedding)
        G.size() -> (#batch, #sentence)
        C.size() -> (#batch, #hidden)
        concat.size() -> (#batch, 3 x #embedding)
        '''
        G = self.make_interaction(facts, questions, prevM)
        value, index = torch.max(G, dim=1)

        #print("focous = ",interpret_indexed_tensor(contexts[:, index[0], :]))
        
        C = self.AGRU(facts, G)
        concat = torch.cat([prevM.squeeze(1), C, questions.squeeze(1)], dim=1)
        next_mem = F.relu(self.next_mem(concat))
        next_mem = next_mem.unsqueeze(1)
        return next_mem


# In[1971]:


def position_encoding(embedded_sentence):
    
    sentence_length = embedded_sentence.size()[2]
    embedding_length = embedded_sentence.size()[3]
    shape = (embedding_length, sentence_length)
    l = np.empty(shape)

    for word_index in range(sentence_length):
        for e_index in range(embedding_length):
            l[e_index][word_index]=(1 - word_index/(sentence_length-1)) - (e_index/(embedding_length-1)) * (1 - 2*word_index/(sentence_length-1))
    l=l.T
    l = torch.FloatTensor(l)
    l = l.unsqueeze(0) # for #batch
    l = l.unsqueeze(1) # for #sen
    print("embedded_sentence.size() = ",embedded_sentence.size())
    print("before ",(l.size()))
    l = l.expand_as(embedded_sentence)
    print("after ",(l.size()))
    weighted = embedded_sentence * Variable(l)
    var = torch.sum(weighted, dim=2).squeeze(2)
    print("return size", var.size())
    return torch.sum(weighted, dim=2).squeeze(2) # sum with tokens


# In[1989]:


class InputModule(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(InputModule, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, bidirectional=True, batch_first=True)
        for name, param in self.gru.state_dict().items():
            if 'weight' in name: init.xavier_normal(param)
        self.dropout = nn.Dropout(0.1)

    def forward(self, contexts, word_embedding):
        print(contexts.size())
        batch_num, sen_num, token_num = contexts.size()

        contexts = contexts.view(batch_num, -1)
        contexts = word_embedding(contexts)

        contexts = contexts.view(batch_num, sen_num, token_num, -1)
        contexts = position_encoding(contexts)
        contexts = self.dropout(contexts)

        h0 = Variable(torch.zeros(2, batch_num, self.hidden_size))
        facts, hdn = self.gru(contexts, h0)
        facts = facts[:, :, :hidden_size] + facts[:, :, hidden_size:]
        return facts


# In[1994]:


memory = EpisodicMemory(hidden_size)


# In[1996]:


Question="who is this"


# In[1997]:


Question=variable_from_sentence("",type_="Q")
Question = Variable(Question.unsqueeze(dim=0))


# In[1998]:


question_module = QuestionModule(vocab_size, hidden_size)


# In[2000]:


questions = question_module(Question, word_embedding)


# In[2003]:



M = questions
for hop in range(3):
    M = memory(f, questions, M)


# In[2008]:


qf_embedding=torch.mul(M,questions)


# In[1992]:


word_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0, sparse=True)
       
input_module = InputModule(vocab_size, hidden_size)
f = input_module(fact_var_tensor, word_embedding)


# In[1965]:


fact_var=[]
for fact in trial_fact:
    fact_var.append(variable_from_sentence(fact,type_="F"))


# In[1985]:


'''
facts_list=[]
for index,row in facts.iterrows():
    facts_list.append([row["entity"],row["relation"],row["attribute"]])
'''


# In[1933]:


f_l=[]
for fact in facts_list:
    f_l.append( " ".join(str(x) for x in fact))


# In[1951]:


fact_words=" ".join(fact for fact in f_l)


# In[2179]:


variables_of_fact=[]
for index,fact in total_facts.iterrows():
    print(index,fact['entity'] +" "+ fact['relation'] +" "+ fact['attribute'])
    variables_of_fact.append(variable_from_sentence(fact['entity'] +" "+ fact['relation'] +" "+ fact['attribute'],type_="F"))
    
    
  
    

