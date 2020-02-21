#!/usr/bin/env python
# coding: utf-8

# This notebook processes Alcohol Language Corpus, and examines the pausing pattern in intoxicated vs. sober speech in German, as examplified in the said corpus. Information about the data can be found at https://www.phonetik.uni-muenchen.de/Bas/BasALCeng.html 

# In[10]:


get_ipython().run_line_magic('matplotlib', 'inline')
import os

os.chdir("/Volumes/HongZhangTOSHIBA/AlcoholLanguageCorpus")

from importlib import reload
import process_alcohol as pa
from scipy import linalg
from matplotlib import pyplot as plt
import seaborn as sns
from glob import glob
import re

# Intoxicated sessions are session 1 and 3, Sober sessions are 2 and 4.
# List all the relevant dir first
reload(pa)
AllDir = next(os.walk('.'))[1]
ADir = [d for d in AllDir if re.match("ses[13]",d)]
NADir = [d for d in AllDir if re.match("ses[24]",d)]

# Read in files from each listed dir
# For ADir:
Aparfiles = {}
AFiles = {}
for A in ADir:
    AF = pa.AlcoholLanguage(A, pa.blocksA)
    AF.list_files()
    AF.read_files()
    AFiles[AF.speaker] = AF
    Aparfiles[AF.speaker] = AF.parfiles
    
# For NADir:
NAparfiles = {}
NAFiles = {}
for NA in NADir:
    NAF = pa.AlcoholLanguage(NA, pa.blocksNA)
    NAF.list_files()
    NAF.read_files()
    NAFiles[NAF.speaker] = NAF
    NAparfiles[NAF.speaker] = NAF.parfiles


# In[11]:


# testing code
#reload(pa)
#ttNA = pa.AlcoholLanguage('ses3015', pa.blocksNA)
#ttNA.list_files()
#ttNA.read_files()
#ttNA.word_stamps
#ttNA.get_repeat()
#ttNA.tokenDic


# # repetitions in alcolhal language

# In[12]:


#[k for k in AFiles]


# In[56]:


rtest = NAFiles[57].get_repeat()
rtest


# In[65]:


NAFiles[57].tokenDic[2][38]


# In[13]:


# Here put keys to repetitions in A and NA categories in two dictionaries
from copy import deepcopy
A_repeats = {}
NA_repeats = {}
for k in [k for k in AFiles]:
    A_repeat_dict = AFiles[k].get_repeat()
    A_repeats[k] = deepcopy(A_repeat_dict)
    
for k in [k for k in NAFiles]:
    NA_repeat_dict = NAFiles[k].get_repeat()
    NA_repeats[k] = deepcopy(NA_repeat_dict)
    
# Then do three things:
# 1. Compare the token distribution of repetitions between A and NA
# 2. Compare the relative frequency of fluent repetitions among all repetitions
# 3. (maybe) add some acoustic analysis in addition to the 2 points above.
# 4. Also consider deletion/repair for this data set


# In[134]:


A_rep_keys


# In[15]:


# 1st, let me get a sense of what the repeated words are in each condition

def uniquify(seq):
    # helper function to remove duplicates in a list
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def concat_rep_str(rep_id_lst,cid,bid,AL_obj):
    # get the repeated words as strings from a list of indices
    word_list = uniquify([AL_obj.tokenDic[cid][j] for j in rep_id_lst])
    return ' '.join(word_list), rep_id_lst[0], bid, cid

A_rep_patterns = []
A_rep_keys = pa.flatten(A_repeats)
for bid,cid in A_rep_keys:
    wids = A_repeats[bid][cid]
    AL_obj = AFiles[bid]
    A_rep_patterns.extend([concat_rep_str(r,cid,bid,AL_obj) for r in wids])
    
# remove 3 repeats:
current_j = -2
current_w = 'none'
A_rep_p = []
A_rep_id = [] # store the id of first repeated word and the total length of repeated segment
three = False
for w,j,bid,cid in A_rep_patterns:
    if w==current_w and j==current_j+1:
        three = True
    else:
        A_rep_p.append(re.sub('/?\+/?','',w))
        if three==False:
            A_rep_id.append((bid,cid,j,len(w.split())))
        three = False
    current_w = w
    current_j = j

############
    
NA_rep_patterns = []
NA_rep_keys = pa.flatten(NA_repeats)
for bid,cid in NA_rep_keys:
    wids = NA_repeats[bid][cid]
    AL_obj = NAFiles[bid]
    NA_rep_patterns.extend([concat_rep_str(r,cid,bid,AL_obj) for r in wids])

current_j = -2
current_w = 'none'
NA_rep_p = []
NA_rep_id = []
three = False
for w,j,bid,cid in NA_rep_patterns:
    if w==current_w and j==current_j+1:
        three = True
    else:
        NA_rep_p.append(re.sub('/?\+/?','',w))
        if three==False:
            NA_rep_id.append((bid,cid,j,len(w.split())))
        three = False
    current_w = w
    current_j = j

# Got to know the total number of tokens in each condition of the corpus
A_corpus = []
for A in AFiles:
    ids = pa.flatten(AFiles[A].words)
    for cid,wid in ids:
        if AFiles[A].words[cid][wid]!='#GARBAGE#':
            A_corpus.append(AFiles[A].words[cid][wid])

NA_corpus = []
for NA in NAFiles:
    ids = pa.flatten(NAFiles[NA].words)
    for cid,wid in ids:
        if NAFiles[NA].words[cid][wid]!='#GARBAGE#':
            NA_corpus.append(NAFiles[NA].words[cid][wid])

print(len(A_corpus))
print(len(NA_corpus))


# In[16]:


# See the frequency of repetitions in each condition:
print(len(NA_rep_patterns)/len(NA_corpus)*1000)
print(len(A_rep_patterns)/len(A_corpus)*1000)


# In[118]:


from collections import Counter

A_rep_counter = Counter([w for w,i1,i2,i3 in A_rep_patterns])
NA_rep_counter = Counter([w for w,i1,i2,i3 in NA_rep_patterns])
#len([k for k in NA_rep_counter if "#" in k])/len(NA_rep_counter)
print(NA_rep_counter.most_common(30))
#len(NA_rep_patterns)


# In[174]:


rep_counter = Counter([w for w,i1,i2,i3 in A_rep_patterns]+[w for w,i1,i2,i3 in NA_rep_patterns])
print(len([k for k in rep_counter if rep_counter[k]>2]))


# In[19]:


# to change the dir to somewhere else, so that I don't need to reload the data again...
os.chdir("/Users/hongzhang/Documents/Mac0docs/disfluency_project/Fisher_extend/individual_var/indvidual_all_words")


# In[17]:


#for bid,cid,wid,j in NA_rep_id:
#    print(bid,cid,wid)
    


# In[74]:


# Find the subclassification of repetitions in each condition here.
# do this mainly by looking at pause durations between repeats.
# The time stamps of words are stored in the property word_stamps of the AlcoholLanguage class

# keys to repeated segments should already be provided in A_rep_id and NA_rep_id.

# then for each tuple in the list, get the start and end times, and group by repeated words
# for A speakers:
from collections import defaultdict

rep_durs_A = []
repid2rowA = defaultdict(list)
row2repidA = {}
row_no = 0
count=0
for bid,cid,j,l in A_rep_id:
    start_r0 = AFiles[bid].word_stamps[cid][j][0]
    end_r0 = AFiles[bid].word_stamps[cid][j+l-1][1]
    if AFiles[bid].words[cid][j+l] in ['<"ah>','<"ahm>','<hm>']:
        try:
            start_r1 = AFiles[bid].word_stamps[cid][j+l+1][0]
            end_r1 = AFiles[bid].word_stamps[cid][j+l+l][1]
        except KeyError:
            # When there is #GARBAGE# between repeated words
            start_r1 = AFiles[bid].word_stamps[cid][j+l+2][0]
            end_r1 = AFiles[bid].word_stamps[cid][j+l+l+1][1]
    else:
        try:
            start_r1 = AFiles[bid].word_stamps[cid][j+l][0]
            end_r1 = AFiles[bid].word_stamps[cid][j+l+l-1][1]
        except KeyError:
            # When there is #GARBAGE# between repeated words
            start_r1 = AFiles[bid].word_stamps[cid][j+l+1][0]
            end_r1 = AFiles[bid].word_stamps[cid][j+l+l][1]
    
    repid2rowA[bid].append(row_no)
    row2repidA[row_no] = bid
    r1 = end_r0-start_r0; r2 = end_r1-start_r1
    #if r2<=0.1:
    #    count+=1
    p = start_r1-end_r0
    if r1>=0.1 or r2>=0.1:
        rep_durs_A.append((r1,p,r2))
        row_no+=1
    else:
        count+=1
print(count)
    
rep_durs_NA = []
repid2rowNA = defaultdict(list)
row2repidNA = {}
row_no = 0
count=0
for bid,cid,j,l in NA_rep_id:
    start_r0 = NAFiles[bid].word_stamps[cid][j][0]
    end_r0 = NAFiles[bid].word_stamps[cid][j+l-1][1]
    if NAFiles[bid].words[cid][j+l] in ['<"ah>','<"ahm>','<hm>','ja']:
        try:
            start_r1 = NAFiles[bid].word_stamps[cid][j+l+1][0]
            end_r1 = NAFiles[bid].word_stamps[cid][j+l+l][1]
        except KeyError:
            # When there is #GARBAGE# between repeated words
            start_r1 = NAFiles[bid].word_stamps[cid][j+l+2][0]
            end_r1 = NAFiles[bid].word_stamps[cid][j+l+l+1][1]
    else:
        try:
            start_r1 = NAFiles[bid].word_stamps[cid][j+l][0]
            end_r1 = NAFiles[bid].word_stamps[cid][j+l+l-1][1]
        except KeyError:
            # When there is #GARBAGE# between repeated words
            start_r1 = NAFiles[bid].word_stamps[cid][j+l+1][0]
            end_r1 = NAFiles[bid].word_stamps[cid][j+l+l][1]
    
    repid2rowNA[bid].append(row_no)
    row2repidNA[row_no] = bid
    r1 = end_r0-start_r0; r2 = end_r1-start_r1
    #if r1<=0.1:
    #    count+=1
        #print(str(bid)+" "+str(cid)+" "+str(j))
    p = start_r1-end_r0
    if r1>=0.1 or r2>=0.1:
        rep_durs_NA.append((r1,p,r2))
        row_no+=1
    else:
        count+=1
print(count)


# In[64]:


np.min(repNAmat[:,1])


# In[78]:


import numpy as np
from matplotlib import pyplot as plt

repAmat = np.array(rep_durs_A)
repNAmat = np.array(rep_durs_NA)

fig, ax = plt.subplots(figsize=(8, 8));
plt.scatter(repNAmat[:,0],repNAmat[:,2],facecolors='none',color="blue",label="Sober");
plt.scatter(repAmat[:,0],repAmat[:,2],facecolors='none',color="red",label="Alcohol");
#ax.set_title("Alcohol",fontsize=22);
#ax[1].set_title("Sober",fontsize=22);
plt.legend(loc="upper right",prop={'size': 20});
ax.set_xlabel("R1 (s)");
ax.set_ylabel("R2 (s)");
ax.set_xlim([0,2.5]);
ax.set_ylim([0,2.5]);


# In[347]:


stats.ttest_ind(repNAmat[:,0],repAmat[:,0],equal_var=False)


# In[348]:


stats.ttest_ind(repNAmat[:,2],repAmat[:,2],equal_var=False)


# In[346]:


stats.ttest_ind(repNAmat[:,0]/repNAmat[:,2],repAmat[:,0]/repAmat[:,2],equal_var=True)


# In[352]:


np.mean(repNAmat[:,0])-np.mean(repAmat[:,0])


# In[132]:


spkr = 544

fig, ax = plt.subplots(figsize=(8, 8));
plt.scatter(repNAmat[repid2rowNA[spkr],0],repNAmat[repid2rowNA[spkr],2],facecolors='none',color="blue",label="Sober");
plt.scatter(repAmat[repid2rowA[spkr],0],repAmat[repid2rowA[spkr],2],facecolors='none',color="red",label="Alcohol");
#ax.set_title("Alcohol",fontsize=22);
#ax[1].set_title("Sober",fontsize=22);
plt.legend(loc="upper right",prop={'size': 20});
ax.set_xlabel("R1 (s)");
ax.set_ylabel("R2 (s)");
ax.set_xlim([0,2.5]);
ax.set_ylim([0,2.5]);


# In[105]:


[k for k in repid2rowA]


# In[94]:


import seaborn as sns
sns.set_style('ticks')

ax = plt.figure()  
#plt.axis([-0.5, 1.5, 0, 2.5])
#sns.kdeplot(np.array(FLp2), shade=False, label='FL');
sns.kdeplot(repAmat[:,1], shade=False, label='A');
sns.kdeplot(repNAmat[:,1], shade=False, label='NA');
plt.xlabel("pause duration (s) in p2");
plt.legend(loc='upper right');


# In[349]:


stats.ttest_ind(repAmat[:,1],repNAmat[:,1])


# #### The duration features do not seem to differ between A and NA, for the whole sample. However, it's more likely that changes can be found on each individual speaker (since alcohol's effect on human behavior is highly conditioned on individual properties...).
# 
# Then the aspects to look at include: the frequency change, the token type change, and duration change, in A and NA conditions.

# In[228]:


# Construct individual feature representation here, to explore the change associated with each individual
# caused by alcohol intoxication.
# For each speaker, include the following: raw frequency of repetition (# of repeated words/total words produced)
# and repetition word vector, a 1-0 vector indicating whether any of the top 20 combined repeated token are present
# in the speech in a particular condition.

# make a mapping between column id and word forms...
top20rep = [k for k in rep_counter if rep_counter[k]>1] # not just top 20, but all those appeared more than twice.
vecid2repwrd = {j:w for j,w in enumerate(top20rep)}
repwrd2vecid = {w:j for j,w in enumerate(top20rep)}
spkr2featrow = {s:j for j,s in enumerate(NA_repeats)}
featrow2spkr = {j:s for j,s in enumerate(NA_repeats)}

spkr_freq_A = np.zeros([len(NA_repeats),1])
spkr_type_A = np.zeros([len(NA_repeats),len(top20rep)])
spkr_freq_NA = np.zeros([len(NA_repeats),1])
spkr_type_NA = np.zeros([len(NA_repeats),len(top20rep)])

for spkr in NA_repeats: # use the rows in NA
    spkr_vec = np.zeros([1,len(top20rep)])
    n_repeats = len(repid2rowNA[spkr])
    spkrwrds = []
    for cid,wid in pa.flatten(NAFiles[spkr].words):
        spkrwrds.append(NAFiles[spkr].words[cid][wid])
    rep_freq = n_repeats/len(spkrwrds)*100
    
    # get the repeated tokens:
    # (I will just do another loop through the repeititon dictionary.... and another....)
    for cid in NA_repeats[spkr]:
        wids = NA_repeats[spkr][cid]
        for r in wids:
            repwrd = concat_rep_str(r,cid,spkr,NAFiles[spkr])[0]
            if repwrd in top20rep:
                spkr_vec[0,repwrd2vecid[repwrd]] = 1
    spkr_freq_NA[spkr2featrow[spkr],:] = rep_freq
    spkr_type_NA[spkr2featrow[spkr],:] = spkr_vec
    
    if spkr in repid2rowA: # if this spkr repeated anything, do the same again here:
        spkr_vec = np.zeros([1,len(top20rep)])
        n_repeats = len(repid2rowA[spkr])
        spkrwrds = []
        for cid,wid in pa.flatten(AFiles[spkr].words):
            spkrwrds.append(AFiles[spkr].words[cid][wid])
        rep_freq = n_repeats/len(spkrwrds)*100
    
        for cid in A_repeats[spkr]:
            wids = A_repeats[spkr][cid]
            for r in wids:
                repwrd = concat_rep_str(r,cid,spkr,AFiles[spkr])[0]
                if repwrd in top20rep:
                    spkr_vec[0,repwrd2vecid[repwrd]] = 1
        spkr_freq_A[spkr2featrow[spkr],:] = rep_freq
        spkr_type_A[spkr2featrow[spkr],:] = spkr_vec
   


# In[324]:


# draw 81 random samples 50 times without repetition to get a sense of on average how different within half-sample
# groups are...
from scipy import stats
def within_compare(mat,num_iter,dist_type=2):
    # mat: matrix like spkr_type_A
    # return: mean norm of the within-matrices over num_iter rounds of comparisons
    results = np.zeros(num_iter)
    for j in range(num_iter):
        include_idx = np.random.choice(mat.shape[0], int(np.floor(mat.shape[0]/2)), replace=False)
        mask = np.zeros(mat.shape[0],dtype=bool)
        mask[include_idx] = True
        matA = mat[mask,:]
        matB = mat[~mask,:]
        dist = np.linalg.norm(matA-matB,dist_type)
        results[j] = dist
    return results,np.std(results)


# In[297]:


# Compare how similar spkr_type_A is to spkr_type_NA:

dist_A_NA = np.linalg.norm(spkr_type_A-spkr_type_NA,2)

# Compare the first half of NA to its second half, and the same with A:
dist_A = np.linalg.norm(spkr_type_A[:81,:]-spkr_type_A[81:,],2)
dist_NA = np.linalg.norm(spkr_type_NA[:81,:]-spkr_type_NA[81:,],2)

#cosine_similarity(spkr_type_A.flatten().reshape([1,25596]),spkr_type_NA.flatten().reshape([1,25596]))

# then need to argue that the distance between A and NA is larger than within comparisons.
print(dist_A_NA,dist_A,dist_NA)


# In[325]:


# do some t-tests to compare the between group difference and within group difference 

num_iter = 50
Adists, stdA = within_compare(spkr_type_A,num_iter,dist_type=2)
NAdists, stdNA = within_compare(spkr_type_NA,num_iter,dist_type=2)

t_A, p_A = stats.ttest_1samp(Adists,dist_A_NA)
t_NA, p_NA = stats.ttest_1samp(NAdists,dist_A_NA)

print(p_A,p_NA)


# In[335]:


# plot the bars and error bars...

labs = ['A vs. NA', 'Non-Alcohol', 'Alcohol']
x_pos = np.arange(3)*2
means = [dist_A_NA, np.mean(NAdists), np.mean(Adists)]
error = [0, stdNA, stdA]

fig, ax = plt.subplots(figsize=(8,8))
ax.bar(x_pos, means, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=5)
ax.set_ylabel('2-Norm distance between token type matrices',fontsize=20)
ax.set_xticks(x_pos)
ax.set_xticklabels(labs,fontsize=20)
ax.yaxis.grid(True)
plt.scatter([2],[8.8],color="black",marker="*")
plt.scatter([4],[4.8],color="black",marker="*")

# Save the figure and show
plt.tight_layout()


# In[267]:


spkr_type_NA[:81,:].flatten().shape


# In[337]:


# Copmute the cosine similarity between A and NA for each individual
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

type_A = csr_matrix(spkr_type_A)
type_NA = csr_matrix(spkr_type_NA)
cosine = cosine_similarity(type_A.todense(),type_NA.todense())
individual_sim = np.diag(cosine)

fig, ax = plt.subplots(figsize=(8,8));
#plt.plot(np.sort(np.diag(corr_mat)));
plt.scatter(np.arange(162),np.sort(individual_sim));
ax.set_ylabel('Cosine similarity',fontsize=20);
ax.set_xlabel('Speaker',fontsize=20);
ax.yaxis.grid(True);
#ax.set_title('Cosine similarity between A and NA',fontsize=20);
#print(np.argsort(individual_sim))


# In[350]:


# See the by-speaker frequency change

fig, ax = plt.subplots(figsize=(8,8));
#plt.plot(np.sort(np.diag(corr_mat)));
cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True, dark=0.5, light=1, reverse=False)
sns.kdeplot(spkr_freq_A.flatten(), spkr_freq_NA.flatten(), cmap=cmap, n_levels=20, shade=True);
plt.scatter(spkr_freq_A, spkr_freq_NA,facecolors='none',color='blue');
ax.set_ylabel('Frequency of NA per 100 words',fontsize=20);
ax.set_xlabel('Frequency of A per 100 words',fontsize=20);
ax.set_xlim([-0.01,2.5]);
ax.set_ylim([-0.01,2.5]);
#ax.set_title('Frequency change between A and NA',fontsize=20);
x = np.linspace(0, 2.5,1000);
plt.plot(x,x,color="red",linewidth=2.0);


# In[20]:


#[k for k in AFiles[28].word_stamps[10]]


# In[110]:


AFiles[10].word_stamps[2][82]


# # Silent pause duration distribution

# In[2]:


# First globally compare two categories: A vs NA
# For each session+block, combine all the silence and speech durations, then do density estimation
import itertools
from collections import defaultdict
from scipy import stats
import numpy as np

spkrs = list(sorted([k for k in AFiles]))

A_sil_spch = {}
for al in spkrs:
    A_sil_spch[al] = list(itertools.chain(*[AFiles[al].pauses[k] for k in AFiles[al].pauses]))

sil_spch_matA = []
for al in A_sil_spch:
    sil, spch = list(zip(*A_sil_spch[al]))
    sil = np.array(sil); spch = np.array(spch)
    
    xmin = sil.min()
    xmax = sil.max()
    ymin = spch.min()
    ymax = spch.max()

    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
        
    values = np.vstack([sil, spch])
    kernel = stats.gaussian_kde(values)
    Z = kernel(positions) #flattened matrix for individual density estimations
    
    sil_spch_matA.append(Z)
    
NA_sil_spch = {}
for al in spkrs:
    NA_sil_spch[al] = list(itertools.chain(*[NAFiles[al].pauses[k] for k in NAFiles[al].pauses]))

sil_spch_matNA = []
for al in NA_sil_spch:
    sil, spch = list(zip(*NA_sil_spch[al]))
    sil = np.array(sil); spch = np.array(spch)
    
    xmin = sil.min()
    xmax = sil.max()
    ymin = spch.min()
    ymax = spch.max()

    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
        
    values = np.vstack([sil, spch])
    kernel = stats.gaussian_kde(values)
    Z = kernel(positions) #flattened matrix for individual density estimations
    
    sil_spch_matNA.append(Z)


# In[6]:


[al for al in NA_sil_spch][0:20]


# In[7]:


sil, spch = list(zip(*NA_sil_spch[568]))
sil = np.array(sil); spch = np.array(spch)
import seaborn as sns; sns.set(color_codes=True)
f, ax = plt.subplots(1,2,figsize=(16, 8))
cmap = sns.cubehelix_palette(rot=-.8, as_cmap=True, dark=0.1, light=1, reverse=True)
sns.kdeplot(spch, sil, cmap=cmap, n_levels=20, shade=True,ax=ax[0]);
ax[0].set_xlim([-0.5,7]);
ax[0].set_ylim([-0.2,2.5])
ax[0].set_xlabel("Following speech duration (s)",fontsize=22);
ax[0].set_ylabel("Silence duration (s)",fontsize=22);
ax[0].set_title("Sober",fontsize=22);

sil, spch = list(zip(*A_sil_spch[568]))
sil = np.array(sil); spch = np.array(spch)
#f, ax = plt.subplots(figsize=(8, 8))
cmap = sns.cubehelix_palette(rot=-.8, as_cmap=True, dark=0.1, light=1, reverse=True)
sns.kdeplot(spch, sil, cmap=cmap, n_levels=20, shade=True,ax=ax[1]);
ax[1].set_xlim([-0.5,7]);
ax[1].set_ylim([-0.2,2.5])
ax[1].set_xlabel("Following speech duration (s)",fontsize=22);
ax[1].set_ylabel("Silence duration (s)",fontsize=22);
ax[1].set_title("Intoxicated",fontsize=22);


# In[9]:


# Then the SVD stuff...
# Alcohol speech:
UfA, sfA, VhfA = linalg.svd(sil_spch_matA)
SfA = np.diag(sfA)
SrfA = SfA[:,:2]
BfA = np.matmul(UfA,SfA)

# Non-alcohol speech:
UfNA, sfNA, VhfNA = linalg.svd(sil_spch_matNA)
SfNA = np.diag(sfNA)
SrfNA = SfNA[:,:2]
BfNA = np.matmul(UfNA,SfNA)


# In[15]:


# Should I concatenate the two matrices then do SVD?

# Then the SVD stuff...
all_spch_mat = np.vstack((sil_spch_matA,sil_spch_matNA))
Ufa, sfa, Vhfa = linalg.svd(all_spch_mat)
Sfa = np.diag(sfa)
Srfa = Sfa[:,:2]
Bfa = np.matmul(Ufa,Srfa)


# In[51]:


fig = plt.figure(figsize=(6,6))
ax = plt.axes([0., 0., 1., 1.])
ax.set_facecolor('white');
ax.grid(b=True, which='major',linestyle="--",color="grey");

s = 100
#plt.scatter(X[:, 0], X[:, 1], color='red', s=s, lw=0, label='True Position')
#for i, txt in enumerate(labels):
#    ax.annotate(txt, (X[i, 0], X[i, 1]), fontsize=15, color="red")

ax.scatter(BfA[:,0], BfA[:,1],facecolors='none',color='red', s=s, lw=1,label='Intoxicated');
ax.scatter(BfNA[:,0], BfNA[:,1],facecolors='none',color='blue', s=s, lw=1,label='Sober');
plt.axvline(x=-9.5, color='black', linestyle='-',lw=0.8)
plt.axhline(y=0.5, color='black', linestyle='-',lw=0.8)
plt.xlabel("First dimension",fontsize=18);
plt.ylabel("Second dimension",fontsize=18);
ax.legend(loc="upper right",prop={'size': 20});


# In[50]:


fig = plt.figure(figsize=(6,6))
ax = plt.axes([0., 0., 1., 1.])
ax.set_facecolor('white')
ax.grid(b=True, which='major',linestyle="--",color="grey");

s = 100
#plt.scatter(X[:, 0], X[:, 1], color='red', s=s, lw=0, label='True Position')
#for i, txt in enumerate(labels):
#    ax.annotate(txt, (X[i, 0], X[i, 1]), fontsize=15, color="red")

ax.scatter(UfA[:,1], BfA[:,2],facecolors='none',color='red', s=s, lw=1,label='Intoxicated');
ax.scatter(UfNA[:,1], BfNA[:,2],facecolors='none',color='blue', s=s, lw=1,label='Sober');
#plt.axvline(x=0.01, color='black', linestyle='-',lw=0.8)
plt.xlabel("Second dimension",fontsize=18);
plt.ylabel("Third dimension",fontsize=18)
ax.legend(loc="upper right",prop={'size': 20});


# In[10]:


get_ipython().run_line_magic('matplotlib', 'notebook')
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('white')
ax.grid(b=True, which='major',linestyle="--",color="grey");

ax.scatter(UfA[:,0], UfA[:,1], UfA[:,2]);
ax.scatter(UfNA[:,0], UfNA[:,1], UfNA[:,2]);



# The plot above shows some sort of separation between the drunk and sober group, between the first and second dimensions, as well as second and third dimensions (below or replot). Thus the pausing pattern in spontaneous speech contains information about change in language ability due to alcohol intoxication. Separate look in the same dimensions by blood alcohol concentration level and gender will be discussed later.

# In this section, it would be nice to color the intoxicated speech with different levels of blood alcohol concentration levels, rather than just binary coding sober and drunk.

# In[54]:


# Plot with color coding of blood alcohol concentration level
# Read in the table mapping speaker id to blood alcohol concentration. Create a dictionary of {spkr: concentration}
# The file lives in /Volume/HongZhangTOSHIBA⁩/AlcoholLanguageCorpus⁩/DOC⁩/IS2011CHALLENGE⁩/SID.txt

spkr_drunk_index = defaultdict(dict)
SID = "/Volumes/HongZhangTOSHIBA/AlcoholLanguageCorpus/DOC/IS2011CHALLENGE/SID.txt"
with open(SID,"r") as sid:
    for line in sid:
        group, spkrid, sex, blood, scrambled, breath = line.split()
        spkr = int(spkrid); blood = float(blood)
        spkr_drunk_index[spkr]['sex'] = sex
        spkr_drunk_index[spkr]['blood'] = blood
        spkr_drunk_index[spkr]['breath'] = breath
        
# plot the distribution of blood alcohol concentration
blood_al = [spkr_drunk_index[spkr]['blood'] for spkr in spkr_drunk_index]

fig = plt.figure(figsize=(8,8));
plt.hist(blood_al);


# use 0.0006 as a threshold (per https://alcohol.stanford.edu/alcohol-drug-info/buzz-buzz/what-bac)

# In[56]:


# color the points in the above scattered plot by intoxication level group
# need to create a mapping from speaker id to row number (A matrix only)

spkr2row = {}; row = 0
for al in A_sil_spch:
    spkr2row[al] = row
    row+=1
    
spkr_drunk_index = {k: v for k, v in spkr_drunk_index.items() if len(v)>0}
    
drunk = np.array([spkr2row[i] for i in spkr2row if i in spkr_drunk_index and spkr_drunk_index[i]['blood']>=0.0006])
sober = np.array([spkr2row[i] for i in spkr2row if i in spkr_drunk_index and spkr_drunk_index[i]['blood']<0.0006])
    
fig = plt.figure(figsize=(8,8))
ax = plt.axes([0., 0., 1., 1.])
ax.set_facecolor('white')
ax.grid(b=True, which='major',linestyle="--",color="grey");


s = 50   
    
ax.scatter(BfA[drunk,0], BfA[drunk,1], facecolors='none',color='red', s=s, lw=1,label='Alcohol');
ax.scatter(BfA[sober,0], BfA[sober,1], facecolors='none',color='orange', s=s, lw=1);
ax.scatter(BfNA[:,0], BfNA[:,1],facecolors='none',color='blue', s=s, lw=1);


# Absolute blood alcohol concentration level doesn't seem to have an effect on the distribution in the derived space.

# In[61]:


spkr2rowna = {}; row = 0
for al in NA_sil_spch:
    spkr2rowna[al] = row
    row+=1

drunk = np.array([spkr2row[i] for i in spkr2row if i in spkr_drunk_index and spkr_drunk_index[i]['blood']>=0.0006 and spkr_drunk_index[i]['sex']=="M"])
#sober = np.array([spkr2row[i] for i in spkr2row if i in spkr_drunk_index and spkr_drunk_index[i]['blood']<0.0009 and spkr_drunk_index[i]['sex']=="M"])
allm = np.array([spkr2rowna[i] for i in spkr2rowna if i in spkr_drunk_index and spkr_drunk_index[i]['sex']=="M"])

fig = plt.figure(figsize=(6,6))
ax = plt.axes([0., 0., 1., 1.])
ax.set_facecolor('white')
ax.grid(b=True, which='major',linestyle="--",color="grey");

s = 50   
    
ax.scatter(BfA[drunk,0], BfA[drunk,1], facecolors='none',color='red', s=s, lw=1,label='More Drunk');
#ax.scatter(UfA[sober,0], UfA[sober,1], facecolors='none',color='orange', s=s, lw=1,label='Less Alcohol');
ax.scatter(BfNA[allm,0], BfNA[allm,1],facecolors='none',color='blue', s=s, lw=1,label='Sober');
plt.xlabel("First dimension");
plt.ylabel("Second dimension")
ax.legend(loc="upper right");
plt.title("Male");


# In[62]:


drunk = np.array([spkr2row[i] for i in spkr2row if i in spkr_drunk_index and spkr_drunk_index[i]['blood']>=0.0006 and spkr_drunk_index[i]['sex']=="F"])
#sober = np.array([spkr2row[i] for i in spkr2row if i in spkr_drunk_index and spkr_drunk_index[i]['blood']<0.0006 and spkr_drunk_index[i]['sex']=="F"])
allm = np.array([spkr2rowna[i] for i in spkr2rowna if i in spkr_drunk_index and spkr_drunk_index[i]['sex']=="F"])

fig = plt.figure(figsize=(6,6))
ax = plt.axes([0., 0., 1., 1.])
ax.set_facecolor('white');
ax.grid(b=True, which='major',linestyle="--",color="grey");

s = 50   
    
ax.scatter(UfA[drunk,0], UfA[drunk,1], facecolors='none',color='red', s=s, lw=1,label='More Drunk');
#ax.scatter(UfA[sober,0], UfA[sober,1], facecolors='none',color='orange', s=s, lw=1,label='Less Alcohol');
ax.scatter(UfNA[allm,0], UfNA[allm,1],facecolors='none',color='blue', s=s, lw=1,label='Sober');
plt.xlabel("First dimension");
#plt.ylabel("Second dimension")
ax.legend(loc="upper right");
plt.title("Female");


# Looking at the same space by gender groups, separation between drunk and sober is clearer among male speakers.

# Here I will see how things change when F0 and rms amplitude information is included in the feature vector.
# First, see how F0 and rms amplitude change as a function of alcohol intoxication, as well as blood alcohol concentration level. Then merging these variables into the feature vector found through pause-speech densities and see how some classification task works.
# The F0 and amplitude will be represeted as the density function of the speech segments looked in previous cells.

# In[56]:


# Functions reading in F0 and rms amplitude values, mapping F0 samples to audio samples, as well as doing the density
# estimation for the two variables.

# Should I only look at the F0 and amplitudes for the whole utterance, or just the words before and after pauses?

def read_f0(f0file):
    # This function should return a list of F0s for the provided input filename.
    f0s = []; rmss = []; totalSample = 0; F0RmsDict = {};
    with open(f0file, "r") as f0:
        for line in f0:
            f,d,rms,cor = line.split()
            f = float(f); rms = float(rms)
            F0RmsDict[totalSample] = (f,rms)
            if f>1e-7:
                f0s.append(f)
                rmss.append(rms)
            totalSample += 1
    return f0s, rmss, totalSample, F0RmsDict

def convert_sample(AudioSample,AudioSampleRate=44100):
    # This function converts the input sample number from the audio file to the sample number in .f0 file
    # Assuming that frame size of f0 files is 10ms, and step size is 5ms
    ConversionRatio = 0.005*AudioSampleEate
    ConvertedSample = int(np.ceil(AudioSample/ConversionRatio))
    return ConvertedSample

# Use the dictionary Aparfiles and NAparfiles to find the mapping from speaker to their .par files
def get_f0_name(parfilename):
    # This function converts .par file name to .f0 file name
    BaseName = os.path.splitext(os.path.basename(parfilename))[0]
    f0FileName = "/Volumes/HongZhangTOSHIBA/AlcoholLanguageCorpus/f0s/"+BaseName+".f0"
    return f0FileName

# Now read in the f0 and rms values for all the speech produced by each speaker
ASpkrF0 = defaultdict(list)
ASpkrRMS = defaultdict(list)
for j in Aparfiles:
    parfiles = Aparfiles[j]
    for par in parfiles:
        f0FileName = get_f0_name(par)
        f0s, rmss, totalSample, F0RmsDict = read_f0(f0FileName)
        ASpkrF0[j].extend(f0s)
        ASpkrRMS[j].extend(rmss)
        
NASpkrF0 = defaultdict(list)
NASpkrRMS = defaultdict(list)
for j in NAparfiles:
    parfiles = NAparfiles[j]
    for par in parfiles:
        f0FileName = get_f0_name(par)
        f0s, rmss, totalSample, F0RmsDict = read_f0(f0FileName)
        NASpkrF0[j].extend(f0s)
        NASpkrRMS[j].extend(rmss)
        


# In[58]:


# here convert F0 measurements in Hz to semitone, based on the 10th percentile of all the F0 values in Hz for
# each speaker. Discoard F0s greater than 350 Hz, and smaller than 60 Hz.

def convert2semitone(spkrid, F0Dict, NAF0Dict):
    # For the given speaker, convert his or her F0 values in Hz to semitone
    spkrF0Hz = F0Dict[spkrid]
    spkrF0Hz = np.array([h for h in spkrF0Hz if h>=60 and h<=350])
    quant = np.percentile(NAF0Dict[spkrid],10)
    semitone = 12*np.log2(spkrF0Hz/quant)
    return semitone


# In[59]:


ASpkrSemi = {}
for j in ASpkrF0:
    ASpkrSemi[j] = convert2semitone(j,ASpkrF0,NASpkrF0)
        
NASpkrSemi = {}
for j in NASpkrF0:
    NASpkrSemi[j] = convert2semitone(j,NASpkrF0,NASpkrF0)

# Here do the density estimation for each speaker
spkr2semirow = {}
semi_matA = []
resampledA = []
semirow = 0
for al in ASpkrSemi:
    semi = ASpkrSemi[al]
    
    xmin = semi.min()
    xmax = semi.max()
    
    X = np.mgrid[xmin:xmax:100j]
    positions = np.vstack([X.ravel()])
        
    #values = np.vstack([sil, spch])
    kernel = stats.gaussian_kde(semi)
    Z = kernel(positions) #flattened matrix for individual density estimations
    resampled = kernel.resample(100)
    resampledA.append(resampled)
    semi_matA.append(Z)
    spkr2semirow[al] = semirow
    semirow+=1

spkr2semirowna = {}
semi_matNA = []
resampledNA = []
semirowna = 0
for al in NASpkrSemi:
    semi = NASpkrSemi[al]
    
    xmin = semi.min()
    xmax = semi.max()
    
    X = np.mgrid[xmin:xmax:100j]
    positions = np.vstack([X.ravel()])
        
    #values = np.vstack([sil, spch])
    kernel = stats.gaussian_kde(semi)
    resampled = kernel.resample(100)
    resampledNA.append(resampled)
    Z = kernel(positions) #flattened matrix for individual density estimations
    
    semi_matNA.append(Z)
    
    spkr2semirowna[al] = semirowna
    semirowna+=1

    


# In[24]:


np.array(semi_matNA).shape


# Fundamental frequency difference between drunk and sober condition, for male and female separately. The female group shows a somewhat bi-modal distribution. In both cases, the drunk condition has higher overall F0 than sober condition.

# In[28]:


import seaborn as sns

allASpkrSemi = list(itertools.chain(*[ASpkrSemi[v] for v in ASpkrSemi if v in spkr_drunk_index and spkr_drunk_index[v]['sex']=="M"]))
allNASpkrSemi = list(itertools.chain(*[NASpkrSemi[v] for v in NASpkrSemi if v in spkr_drunk_index and spkr_drunk_index[v]['sex']=="M"]))

fig = plt.figure(figsize=(6,6))
sns.kdeplot(allASpkrSemi, shade=False, label='alcohol');
sns.kdeplot(allNASpkrSemi, shade=False, label='sober');
plt.xlim((-5,13));
plt.xlabel("semitone");
plt.title("Male");


# In[29]:


allASpkrSemi = list(itertools.chain(*[ASpkrSemi[v] for v in ASpkrSemi if v in spkr_drunk_index and spkr_drunk_index[v]['sex']=="F"]))
allNASpkrSemi = list(itertools.chain(*[NASpkrSemi[v] for v in NASpkrSemi if v in spkr_drunk_index and spkr_drunk_index[v]['sex']=="F"]))

fig = plt.figure(figsize=(6,6))
sns.kdeplot(allASpkrSemi, shade=False, label='alcohol');
sns.kdeplot(allNASpkrSemi, shade=False, label='sober');
plt.xlim((-10,20));
plt.xlabel("semitone");
plt.title("Female");


# Now look at RMS amplitude

# In[66]:


allASpkrRMS = list(itertools.chain(*[ASpkrRMS[v] for v in ASpkrRMS if v in spkr_drunk_index and spkr_drunk_index[v]['sex']=="M"]))
allNASpkrRMS = list(itertools.chain(*[NASpkrRMS[v] for v in NASpkrRMS if v in spkr_drunk_index and spkr_drunk_index[v]['sex']=="M"]))

sns.kdeplot(allASpkrRMS, shade=False, label='alcohol');
sns.kdeplot(allNASpkrRMS, shade=False, label='sober');
plt.xlim((-100,2000));
plt.xlabel("RMS Amplitude");
plt.title("Male");


# In[67]:


allASpkrRMS = list(itertools.chain(*[ASpkrRMS[v] for v in ASpkrRMS if v in spkr_drunk_index and spkr_drunk_index[v]['sex']=="F"]))
allNASpkrRMS = list(itertools.chain(*[NASpkrRMS[v] for v in NASpkrRMS if v in spkr_drunk_index and spkr_drunk_index[v]['sex']=="F"]))

sns.kdeplot(allASpkrRMS, shade=False, label='alcohol');
sns.kdeplot(allNASpkrRMS, shade=False, label='sober');
plt.xlim((-100,2000));
plt.xlabel("RMS Amplitude");
plt.title("Female");


# Combining F0 and pausing as the feature vector and see what happens

# In[87]:


# Maybe rearrange the feature vectors for decomposition

combo_matA = []
for al in A_sil_spch:
    sil, spch = list(zip(*A_sil_spch[al]))
    sil = np.array(sil); spch = np.array(spch)
    
    xmin = sil.min()
    xmax = sil.max()
    ymin = spch.min()
    ymax = spch.max()

    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
        
    values = np.vstack([sil, spch])
    kernel = stats.gaussian_kde(values)
    Zd = kernel(positions) #flattened matrix for individual density estimations
    
    f0 = np.array(ASpkrF0[al])
    
    xmin = f0.min()
    xmax = f0.max()
    
    X = np.mgrid[xmin:xmax:1000j]
    positions = np.vstack([X.ravel()])
        
    #values = np.vstack([sil, spch])
    kernel = stats.gaussian_kde(f0)
    Zs = kernel(positions)
    #resampled = kde.sample(N_POINTS_RESAMPLE)
    
    Z = np.concatenate([Zd,Zs])
    
    combo_matA.append(Z)
    
combo_matNA = []
for al in NA_sil_spch:
    sil, spch = list(zip(*NA_sil_spch[al]))
    sil = np.array(sil); spch = np.array(spch)
    
    xmin = sil.min()
    xmax = sil.max()
    ymin = spch.min()
    ymax = spch.max()

    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
        
    values = np.vstack([sil, spch])
    kernel = stats.gaussian_kde(values)
    Zd = kernel(positions) #flattened matrix for individual density estimations
    
    f0 = np.array(NASpkrF0[al])
    
    xmin = f0.min()
    xmax = f0.max()
    
    X = np.mgrid[xmin:xmax:1000j]
    positions = np.vstack([X.ravel()])
        
    #values = np.vstack([sil, spch])
    kernel = stats.gaussian_kde(f0)
    Zs = kernel(positions) 
    
    Z = np.concatenate([Zd,Zs])
    
    combo_matNA.append(Z)



# In[89]:


# Then the SVD stuff...
# Alcohol speech:
UfAc, sfAc, VhfAc = linalg.svd(combo_matA)
SfAc = np.diag(sfAc)
SrfAc = SfAc[:,:2]
BfAc = np.matmul(UfAc,SrfAc)

# Non-alcohol speech:
UfNAc, sfNAc, VhfNAc = linalg.svd(combo_matNA)
SfNAc = np.diag(sfNAc)
SrfNAc = SfNAc[:,:2]
BfNAc = np.matmul(UfNAc,SrfNAc)


# In[23]:


#fig = plt.figure(figsize=(8,8))
#ax = plt.axes([0., 0., 1., 1.])

#s = 50
#plt.scatter(X[:, 0], X[:, 1], color='red', s=s, lw=0, label='True Position')
#for i, txt in enumerate(labels):
#    ax.annotate(txt, (X[i, 0], X[i, 1]), fontsize=15, color="red")

#ax.scatter(UfAc[:,-2], UfAc[:,-1],facecolors='none',color='red', s=s, lw=1,label='Alcohol');
#ax.scatter(UfNAc[:,-2], UfNAc[:,-1],facecolors='none',color='blue', s=s, lw=1,label='Sober');
#plt.xlabel("First dimension");
#plt.ylabel("Second dimension")
#ax.legend(loc="upper right");


# In[68]:


len(allASpkrRMS)


# Not seeing much improvement--maybe should go straight to use combined SVD and F0 features to run some classification tests... Try SVM then.

# In[63]:


# SVM for classification into drunk and sober group with joint pausing and F0 features

# TODO:
# Properly perform this classification task: perform classification multiple times, and proper plotting the performance
# as a function of the amount of data used.
# Also try logistic regression, see how things gonig.

import sklearn
import sys
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,f1_score
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF

def train_test_split(spkr2row,UfA,UfNA,partition=1,seed=10):
    np.random.seed(seed)
    # Generate training set. Hard coded in the split ration for this problem...
    speakers = [k for k in spkr2row]
    TrainID = np.random.choice(speakers,122,replace=False)
    TestID = [k for k in speakers if k not in TrainID]

    TrainX = np.zeros([244,162]) #pre-allocate the matrix.
    
    # Adding F0 features doesn't seem to improve the performance of the model, so not included
    for j,k in enumerate(TrainID[:122]):
        row = spkr2row[k]
        #semirow = spkr2semirow[k]
        #semirowna = spkr2semirowna[k]
        TrainX[j,:] = UfA[row,:]
        #TrainX[j,100:] = resampledA[semirow]
        TrainX[j+122,:] = UfNA[row,:]
        #TrainX[j+122,100:] = resampledNA[semirowna]
    
    TestX = np.zeros([80,162]) 

    for j,k in enumerate(TestID):
        row = spkr2row[k]
        #semirow = spkr2semirow[k]
        #semirowna = spkr2semirowna[k]
        TestX[j,:] = UfA[row,:]
        #TestX[j,100:] = resampledA[semirow]
        TestX[j+40,:] = UfNA[row,:]
        #TestX[j+40,100:] = resampledNA[semirowna]
    
    TrainY = np.concatenate([np.ones(122),np.zeros(122)])
    TestY = np.concatenate([np.zeros(40),np.ones(40)])
    
    return TrainX, TrainY, TestX, TestY

_gaussSigma = 1

def myGaussianKernel(X1, X2):
    '''
        Arguments:
            X1 - an n1-by-d numpy array of instances
            X2 - an n2-by-d numpy array of instances
        Returns:
            An n1-by-n2 numpy array representing the Kernel (Gram) matrix
    '''
    n1,d = X1.shape; n2,d = X2.shape
    
    X1ss = np.sum(np.square(X1),axis=1)
    X2ss = np.sum(np.square(X2),axis=1)
    crossproduct = 2*X1.dot(X2.T)
    XX1 = np.array([X1ss,]*n2).T; XX2 = np.array([X2ss,]*n1)
    XX12 = XX1 + XX2 - crossproduct
    gauss = np.exp(-XX12/(2*_gaussSigma**2))
    
    return gauss
# Doing a SVM with gaussian kernel using grid search to find the best tuning parameter
# Code borrowed from CIS519 homework 2
Sigma = [1./(2*sigma**2) for sigma in np.linspace(500,1000, num=100)]
class GaussKernel(BaseEstimator,TransformerMixin):
    def __init__(self, gamma=1.0):
        super(GaussKernel,self).__init__()
        self.gamma = gamma

    def transform(self, X):
        return myGaussianKernel(X[:,:], self.X_train_[:,:])

    def fit(self, X, y=None, **fit_params):
        self.X_train_ = X
        return self

print('python: {}'.format(sys.version))
print('numpy: {}'.format(np.__version__))
print('sklearn: {}'.format(sklearn.__version__))
#np.random.seed(0)



# In[67]:


# Create a pipeline where our custom predefined kernel Chi2Kernel
# is run before SVC.
pipe = Pipeline([
    ('gauss', GaussKernel()),
    ('svm', SVC()),
])

# Set the parameter 'gamma' of our custom kernel by
# using the 'estimator__param' syntax.
cv_params = dict([
    ('gauss__gamma', Sigma),
    ('svm__kernel', ['rbf']),
    ('svm__C', [1e-11,1e-9,1e-7,1e-5]),
])

# Do grid search to get the best parameter value of 'gamma'.
# Do this 50 times to see how model performance vary across trials:
acc_summary = []
for sd in np.arange(50):
    TrainX, TrainY, TestX, TestY = train_test_split(spkr2row,UfA,UfNA,seed=sd)

    model = GridSearchCV(pipe, cv_params, cv=10, verbose=1, n_jobs=-1)
    model.fit(TrainX, TrainY)
    y_pred = model.predict(TestX)
    acc_test = accuracy_score(TestY, y_pred)
    acc_summary.append(acc_test)

    print("Test accuracy: {}".format(acc_test))
    print("Best params:")
    print(model.best_params_)


# In[69]:


print("mean accuracy: {}".format(np.mean(acc_summary)))
print("standard deviation: {}".format(np.std(acc_summary)))


# percentage training data vs. test performance
# 10%: 0.5375
# 20%: 0.65
# 30%: 0.625
# 40%: 0.7125
# 50%: 0.7625
# 60%: 0.8875
# 70%: 0.675
# 80%: 0.7375
# 90%: 0.925
# 100%: 0.9375

# In[ ]:





# In[ ]:





# In[ ]:





# In[21]:


len(convert2semitone(595,ASpkrF0))


# The results above don't look so bad--maybe I can do a classification task just to see how, using this simple measurement of silence and speech durations along, can classify speaker states...

# In[93]:


# Try using multidimensional scaling to explore the distance between two groups:

from matplotlib.collections import LineCollection

from sklearn.metrics import euclidean_distances
from sklearn.manifold import MDS
from sklearn.decomposition import PCA

dens_diff = np.array([sil_spch_matA[j]-sil_spch_matNA[j] for j in range(162)])
all_sil_spch = np.concatenate((sil_spch_matA,sil_spch_matNA),axis=0)

seed = np.random.RandomState(seed=3)

similarities = euclidean_distances(all_sil_spch)
mds = MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=seed,
                   dissimilarity="precomputed", n_jobs=1)
pos = mds.fit(similarities).embedding_

pos *= np.sqrt((all_sil_spch ** 2).sum()) / np.sqrt((pos ** 2).sum())
clf = PCA(n_components=2)
X = clf.fit_transform(all_sil_spch)
pos = clf.fit_transform(pos)

fig = plt.figure(figsize=(12,12))
ax = plt.axes([0., 0., 1., 1.])

s = 50
#plt.scatter(X[:, 0], X[:, 1], color='red', s=s, lw=0, label='True Position')
#for i, txt in enumerate(labels):
#    ax.annotate(txt, (X[i, 0], X[i, 1]), fontsize=15, color="red")

ax.scatter(pos[:162,0], pos[:162,1],facecolors='none',color='blue', s=s, lw=1, label='MDS')
ax.scatter(pos[162:,0], pos[162:,1],facecolors='none',color='red', s=s, lw=1)

labels = list(range(162)) #'mid-short pause', 
#for i, txt in enumerate(labels):
#    ax.annotate(str(txt), (pos[i, 0], pos[i, 1]), fontsize=5)
#for i, txt in enumerate(labels):
#    ax.annotate(str(txt), (pos[162+i, 0], pos[162+i, 1]), fontsize=5)

plt.xlabel("First principle coordinate");
plt.ylabel("Second principle coordinate");


# In[67]:





# In[56]:


# Maybe a good thing to do is to see how each individual changes in this space......
# Order the points by speaker
# And find the relation between such plots and blood alcohol concentration.



# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




