# -*- coding: utf-8 -*-
"""
Preprocessing script for programmable web dataset.

Extracts data from 20 top api categories that belong to at least one mashup.
Generates various data formats from provided api.csv and mashup.csv.
--- needed: replace w2v_pretrained_vectors with a location containing pretrained w2v vectors

"""

import numpy as np
import collections
from numpy import savetxt, loadtxt
from csv import DictReader, reader, register_dialect, list_dialects
from gensim.models import KeyedVectors
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import LatentDirichletAllocation


# ---------------------------------- file loading -----------------------------------

path = "./"
apifile = "api.csv"
mashupfile = "mashup.csv"
w2v_pretrained_vectors = '../../word2vec/GoogleNews-vectors-negative300.bin'

with open(path + mashupfile) as file:
    reader1 = reader(file)
    mashup_doc_l = []
    for row in reader1:
        mashup_doc_l.append(row)
print(len(mashup_doc_l), 'mashup lines read')

with open(path + apifile) as file2:
    reader2 = reader(file2)
    api_doc_l = []
    for row in reader2:
        api_doc_l.append(row)
print(len(api_doc_l), 'api lines read')

# initializes main data structures
doc_l_description = list()
doc_l_target = list()
doc_l_title = list()
doc_l_provider = list()
doc_l_tags = list()
doc_l_mashupnames = [list() for i in range(len(api_doc_l))]

mashup_l_title = list()
mashup_l_refs = list()
mashup_l_description = list()
mashup_l_tags = list()
mashup_l_tagnames = list()

tag_l_title = list()
provider_l_title = list()

rel_dict = dict()  # mashup-service dictionary, each mashup name is a key for a set of service names
service_id_dict = dict()
mashup_mid_dict = dict()

tag_doc_dict = dict()
tag_tagid_dict = dict()

tag_mashup_dict = dict()
mtag_mtagid_dict = dict()

provider_doc_dict = dict()
provider_pid_dict = dict()

# loads mashup data
mashup_count = 0
for p in mashup_doc_l:
    # 1 = title, x = name, x = label, x = summary, 3 = description, 2 = tags, 5 = apis
    if p[1] != '':
        mashup_l_title.append(p[1])
        reflist = list(p[5].split("$"))
        reflist2 = list()
        for i, r in enumerate(reflist):
            reflist2.append(r)
        mashup_l_refs.append(reflist2)
        mashup_l_description.append(p[3] + " " + p[1])
        mashup_l_tagnames.append(' '.join(p[2].split("$")))
        mashup_l_tags.append(list(p[2].split("$")))
        mashup_count += 1

# loads service data
service_count = 0
for p in api_doc_l:
    # 3 = description, 1 = title, x = summary, 5 = tags, 4 = category, 8 = provider(endpoint), 2 = category+tags
    if p[1] != '':
        doc_l_title.append(p[1])
        doc_l_description.append(p[3] + " " + p[1])
        doc_l_target.append(p[4])
        doc_l_tags.append(list(p[5].split("$")))
        service_id_dict[p[1]] = service_count
        service_count += 1
        doc_l_provider.append(p[8])

# builds service-mashup relationship dictionary from mashup data
# in mashup-service dictionary, each mashup name is a key for a set of service names
for i, reflist in enumerate(mashup_l_refs):
    for j, r in enumerate(reflist):
        for r2 in reflist:
            if (r in rel_dict.keys()):
                rel_dict[r].add(r2)
            else:
                rel_dict[r] = set()
                rel_dict[r].add(r2)

# builds service-mashup matrix
x_mashups = np.zeros((service_count, mashup_count), dtype=int)
mashup_id = 0
for p in mashup_doc_l:
    # mashup: p[1] = title, p[4] = name, p[5] = label, p[2] = summary, p[7] = description, p[16] = apis
    if p[1] != '':
        for s in mashup_l_refs[mashup_id]:
            s = s.strip()
            if s + ' API' in service_id_dict.keys():
                x_mashups[service_id_dict[s + ' API'], mashup_id] = 1
                doc_l_mashupnames[service_id_dict[s + ' API']].append(p[1])
        mashup_mid_dict[mashup_l_title[mashup_id]] = mashup_id
        mashup_id += 1
doc_l_nmashups = np.sum(x_mashups, 1)
print(mashup_id, 'mashups processed')



# ------------------------ mashup-tag matrix construction -------------------------

for i, taglist in enumerate(mashup_l_tags):
    for tag in taglist:
        if (tag in tag_mashup_dict):
            tag_mashup_dict[tag].add(i)
        elif tag != '':
            tag_mashup_dict[tag] = set()
            tag_mashup_dict[tag].add(i)

# assigns a mtag id for each tag in the mashup dictionary
tid = 0
for tag in tag_mashup_dict.keys():
    mtag_mtagid_dict[tag] = tid
    tid += 1

# builds mashup-tag matrix
mtag_count = len(tag_mashup_dict.keys())
x_mtags = np.zeros((mashup_count, mtag_count), dtype=int)
mtag_id = 0
for tag in tag_mashup_dict.keys():
    for mashup in tag_mashup_dict[tag]:
        x_mtags[mashup, mtag_mtagid_dict[tag]] = 1
    mtag_id += 1
print(mtag_id, 'mashup tags processed')



# ------------------------ service-tag matrix construction -------------------------

for i, taglist in enumerate(doc_l_tags):
    for tag in taglist:
        if (tag in tag_doc_dict):
            tag_doc_dict[tag].add(i)
        else:
            tag_doc_dict[tag] = set()
            tag_doc_dict[tag].add(i)

# assigns a tag id for each tag in the tag_doc dictionary
tid = 0
for tag in tag_doc_dict.keys():
    tag_tagid_dict[tag] = tid
    tag_l_title.append(tag)
    tid += 1

# builds service-tag matrix
tag_count = len(tag_doc_dict.keys())
x_tags = np.zeros((service_count, tag_count), dtype=int)
tag_id = 0
for tag in tag_doc_dict.keys():
    for service in tag_doc_dict[tag]:
        # if service in service_id_dict.keys():
        x_tags[service, tag_tagid_dict[tag]] = 1
        # print ('x')
    tag_id += 1
print(tag_id, 'api tags processed')


# ------------------------------ service-provider matrix construction ----------------------------

for i, provider in enumerate(doc_l_provider):
    if provider in provider_doc_dict:
        provider_doc_dict[provider].add(i)
    elif provider != '':
        provider_doc_dict[provider] = set()
        provider_doc_dict[provider].add(i)

# assigns a provider id for each provider in the provider_doc dictionary
pid = 0
for provider in provider_doc_dict.keys():
    provider_pid_dict[provider] = pid
    provider_l_title.append(provider)
    pid += 1

# builds service-provider matrix
provider_count = len(provider_doc_dict.keys())
x_providers = np.zeros((service_count, provider_count), dtype=int)
pid = 0
for provider in provider_doc_dict.keys():
    for service in provider_doc_dict[provider]:
        x_providers[service, provider_pid_dict[provider]] = 1
    pid += 1
print(pid, 'providers processed')


# ----------------------- target (api category) and target id processing ------------------------
# creates a target dictionary to calculate target ids and a list of target ids
target_dict = dict()
doc_l_tid = list()
tid = 0
target_counter = collections.Counter()
for i, t in enumerate(doc_l_target):
    if doc_l_nmashups[i] > 0:
        target_counter[t] += 1

# finds top k targets
k = 20
top_targets = list()
for i in range(k):
    current_target = '!'
    current_frequency = 0
    for target in target_counter.keys():
        if target_counter[target] > current_frequency and target not in top_targets and target != '':
            current_target = target
            current_frequency = target_counter[target]
    top_targets.append(current_target)

# creates a tag dictionary to count the frequency of tags over the dataset
tag_counter = collections.Counter()
for tag_list in doc_l_tags:
    for tag in tag_list:
        tag_counter[tag] += 1




# -------------- low/high frequency category-based doc filter (balancing)  ------------------

# sets minimum and maximum number of services per cluster, and minimum mashup refs in service
min_pop = 1
max_pop = 50000
min_refs = 1

doc_l_description_2 = [doc_l_description[i] for i in range(len(doc_l_target)) if
                       target_counter[doc_l_target[i]] > min_pop and doc_l_nmashups[i] > min_refs and doc_l_target[
                           i] in top_targets]
doc_l_title_2 = [doc_l_title[i] for i in range(len(doc_l_target)) if
                 target_counter[doc_l_target[i]] > min_pop and doc_l_nmashups[i] > min_refs and doc_l_target[
                     i] in top_targets]
doc_l_target_2 = [doc_l_target[i] for i in range(len(doc_l_target)) if
                  target_counter[doc_l_target[i]] > min_pop and doc_l_nmashups[i] > min_refs and doc_l_target[
                      i] in top_targets]
doc_l_did_2 = [i for i in range(len(doc_l_target)) if
               target_counter[doc_l_target[i]] > min_pop and doc_l_nmashups[i] > min_refs and doc_l_target[
                   i] in top_targets]
doc_l_tags_2 = [x_tags[i] for i in range(len(doc_l_target)) if
                target_counter[doc_l_target[i]] > min_pop and doc_l_nmashups[i] > min_refs and doc_l_target[
                    i] in top_targets]
doc_l_mashup_2 = [x_mashups[i] for i in range(len(doc_l_description)) if
                  target_counter[doc_l_target[i]] > min_pop and doc_l_nmashups[i] > min_refs and doc_l_target[
                      i] in top_targets]
doc_l_tagnames_2 = [' '.join(doc_l_tags[i]) for i in range(len(doc_l_target)) if
                    target_counter[doc_l_target[i]] > min_pop and doc_l_nmashups[i] > min_refs and doc_l_target[
                        i] in top_targets]
doc_l_mashupnames_2 = ['$'.join(doc_l_mashupnames[i]) for i in range(len(doc_l_target)) if
                       target_counter[doc_l_target[i]] > min_pop and doc_l_nmashups[i] > min_refs and doc_l_target[
                           i] in top_targets]
doc_l_providers_2 = [x_providers[i] for i in range(len(doc_l_target)) if
                     target_counter[doc_l_target[i]] > min_pop and doc_l_nmashups[i] > min_refs and doc_l_target[
                         i] in top_targets]

# high population filter: limits the number of elements of each target to max_pop
cluster_counter = collections.Counter()
doc_l_description_3 = list()
doc_l_title_3 = list()
doc_l_target_3 = list()
doc_l_did_3 = list()
doc_l_tags_3 = list()
doc_l_mashup_3 = list()
doc_l_tagnames_3 = list()
doc_l_mashupnames_3 = list()
doc_l_providers_3 = list()
for i, desc in enumerate(doc_l_description_2):
    cluster_counter[doc_l_target_2[i]] += 1
    if cluster_counter[doc_l_target_2[i]] <= max_pop:
        doc_l_description_3.append(desc)
        doc_l_title_3.append(doc_l_title_2[i])
        doc_l_target_3.append(doc_l_target_2[i])
        doc_l_did_3.append(doc_l_did_2[i])
        doc_l_tags_3.append(doc_l_tags_2[i])
        doc_l_mashup_3.append(doc_l_mashup_2[i])
        doc_l_tagnames_3.append(doc_l_tagnames_2[i])
        doc_l_mashupnames_3.append(doc_l_mashupnames_2[i])
        doc_l_providers_3.append((doc_l_providers_2[i]))

# calculate target ids
for t in doc_l_target_3:
    if t in target_dict.keys():
        doc_l_tid.append(target_dict.get(t))
    else:
        doc_l_tid.append(tid)
        target_dict[t] = tid
        tid += 1

print(len(doc_l_description_3), 'services processed (top-20 categories, only services on mashups, unbalanced)')



# ----------------------- tfidf / lda / word2vec / sine -----------------------

# w2v
vectors = KeyedVectors.load_word2vec_format(w2v_pretrained_vectors, binary=True)
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(doc_l_description_3)]
X_w = list()
counts = list()
for doc in documents:
    vec_list = list()
    count = 0
    for word in doc.words.split():
        if word in vectors.vocab:
            vec_list.append(vectors[word])
        count += 1
    vec = np.mean(vec_list, 0)
    X_w.append(vec)
    counts.append(count)
mdocuments = [TaggedDocument(doc, [i]) for i, doc in enumerate(mashup_l_description)]
print('vectorized apis into W2V spaces')
MX_w = list()
counts = list()
for doc in mdocuments:
    vec_list = list()
    count = 0
    for word in doc.words.split():
        if word in vectors.vocab:
            vec_list.append(vectors[word])
        count += 1
    vec = np.mean(vec_list, 0)
    MX_w.append(vec)
    counts.append(count)
print('vectorized mashups into W2V spaces')

# tfidf
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.9, min_df=2, stop_words=ENGLISH_STOP_WORDS)
vectorizer.fit(doc_l_description_3)
X = vectorizer.transform(doc_l_description_3)
X_t = X.toarray()  # necessary to save X to file since X is a sparse array
print('vectorized apis into a TFIDF space')

# lda - api
vectorizer = CountVectorizer(max_df=1.0, min_df=1, stop_words=ENGLISH_STOP_WORDS)
vectorizer.fit(doc_l_description_3)
pre_X = vectorizer.transform(doc_l_description_3)
lda = LatentDirichletAllocation(n_components=20, random_state=0)
lda.fit(pre_X)
X_l = lda.transform(pre_X)
print('vectorized apis into a LDA space')

# lda - mashup
vectorizer = CountVectorizer(max_df=0.9, min_df=2, stop_words=ENGLISH_STOP_WORDS)
vectorizer.fit(mashup_l_description)
pre_MX = vectorizer.transform(mashup_l_description)
lda = LatentDirichletAllocation(n_components=120, random_state=0)
lda.fit(pre_MX)
MX_l = lda.transform(pre_MX)
print('vectorized mashups into a LDA space')

# sine
t = np.asarray(doc_l_tags_3)
ta = t.dot(t.T)
ta[np.nonzero(ta)] = 1
ta_list = [(i, j, 1) for i in range(386) for j in range(386) if ta[i, j] == 1 and i != j]
m = np.asarray(doc_l_mashup_3)
ma = m.dot(m.T)
ma[np.nonzero(ma)] = 1
ma_list = [(i, j, -1) for i in range(386) for j in range(386) if ma[i, j] == 1 and i != j]
sine_list = ta_list + ma_list


# # -------------------- saves to files ----------------------
#
# savetxt('pw1_386_graph.dat', sine_list, delimiter=' ')
# print('SINE GRAPH dataset saved in matlab .dat format')
#
# savetxt('pw1_386_api_w2v.dat', X_w, delimiter=' ')
# print('API-W2V300 dataset saved in matlab .dat format')
#
# savetxt('pw1_386_api_tfidf.dat', X_t, delimiter=' ')
# print('API-TFIDF dataset saved in matlab .dat format')
#
# savetxt('../pw1_386_api_lda20.dat', X_l, delimiter=' ')
# print('API-LDA dataset saved in matlab .dat format')
#
# savetxt('pw1_386_api_mashup.dat', doc_l_mashup_3, delimiter=' ')
# print('API-MASHUP dataset saved in matlab .dat format')
#
# savetxt('pw1_386_api_provider.dat', doc_l_providers_3, delimiter=' ')
# print('API-PROVIDERS dataset saved in matlab .dat format')
#
# savetxt('pw1_386_api_tag.dat', doc_l_tags_3, delimiter=' ')
# print('API-TAGS FEATURES dataset saved in matlab .dat format')
#
# savetxt('pw1_386_api_target.dat', doc_l_tid, delimiter=' ')
# print('API-TARGET dataset saved in matlab .dat format')
#
# file1 = open("pw1_386_api_names.dat", "w")
# file1.write('\n'.join(doc_l_title_3))
# file1.write('\n')
# file1.close()
# print('API-NAMES dataset saved in matlab .dat format')
#
# file1 = open("pw1_386_api_descriptions.dat", "w")
# file1.write('\n'.join(doc_l_description_3))
# file1.write('\n')
# file1.close()
# print('API-DESCRIPTIONS dataset saved in matlab .dat format')

# file1 = open("pw1_386_api_tag_names.dat", "w")
# file1.write('\n'.join(doc_l_tagnames_3))
# file1.write('\n')
# file1.close()
# print('API-TAGS-NAMES dataset saved in matlab .dat format')
#
# file1 = open("pw1_386_api_mashup_names.dat", "w")
# file1.write('\n'.join(doc_l_mashupnames_3))
# file1.write('\n')
# file1.close()
# print('API-MASHUP-NAMES dataset saved in matlab .dat format')
#
# savetxt('pw1_6366_mashup_tag.dat', x_mtags, delimiter=' ')
# print('MASHUP-TAGS dataset saved in matlab .dat format')
#
# savetxt('pw1_6366_mashup_w2v.dat', MX_w, delimiter=' ')
# print('MASHUP-W2V300 dataset saved in matlab .dat format')
#
# savetxt('pw1_6366_mashup_lda2.dat', MX_l, delimiter=' ')
# print('MASHUP-LDA dataset saved in matlab .dat format')
#
# file1 = open("pw1_6366_mashup_tag_names.dat", "w")
# file1.write('\n'.join(mashup_l_tagnames))
# file1.write('\n')
# file1.close()
# print('MASHUP-TAGS-NAMES dataset saved in matlab .dat format')
#
# file1 = open("pw1_6366_mashup_names.dat", "w")
# file1.write('\n'.join(mashup_l_title))
# file1.write('\n')
# file1.close()
# print('MASHUP-NAMES dataset saved in matlab .dat format')
#
# file1 = open("pw1_386_api_target_names.dat", "w")
# file1.write('\n'.join(doc_l_target_3))
# file1.write('\n')
# file1.close()
# print('API-TARGET-NAMES dataset saved in matlab .dat format')
#
# file1 = open("pw1_477_tag_names.dat", "w")
# file1.write('\n'.join(tag_l_title))
# file1.write('\n')
# file1.close()
# print('TAG-NAMES dataset saved in matlab .dat format')
#
# file1 = open("pw1_4894_provider_names.dat", "w")
# file1.write('\n'.join(provider_l_title))
# file1.write('\n')
# file1.close()
# print('PROVIDER-NAMES dataset saved in matlab .dat format')
#