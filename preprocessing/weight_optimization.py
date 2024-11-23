# -*- coding: utf-8 -*-
"""
weight learning
- feature space weight learning
- metapath weight learning

"""
import numpy as np
from scipy.optimize import minimize


# beta optimization

#1 prep x_k, k=1...5, mask for x and gnd
# x_k = feature space k
x1 = np.asarray(X_w)
x2 = np.asarray(doc_l_tags_3)
x3 = np.asarray(doc_l_mashup_3)
x4 = X_t
x5 = np.asarray(doc_l_providers_3)
gnd = np.asarray(doc_l_tid)

#2 calc c based on gnd (c is a similarity matrix calculated with 2 loop (pair i,j)
c = np.zeros([386, 386])
for i in range(gnd.shape[0]):
    for j in range(gnd.shape[0]):
        if gnd[i] == gnd[j]:
            c[i, j] = 1
        else:
            c[i, j] = -1

# 3 for metapaths
# generate metapaths
m = np.zeros([6, 386, 386])
s = np.zeros([6, 386, 386])
SM = x3  # service-mashup
MS = SM.T
MTm = x_mtags  # mashup-mtag
TmM = MTm.T
STs = x2  # service-tag
TsS = STs.T
SPs = x5  # service-provider
PsS = SPs.T
m[0, :, :] = SM.dot(MS).dot(SM).dot(MS)
m[1, :, :] = SM.dot(MTm).dot(TmM).dot(MS)
m[2, :, :] = SM.dot(MS).dot(STs).dot(TsS).dot(SM).dot(MS)
m[3, :, :] = SM.dot(MS).dot(SPs).dot(PsS).dot(SM).dot(MS)
m[4, :, :] = SPs.dot(PsS)
m[5, :, :] = STs.dot(TsS)
s[0, :, :] = abs(c) * m[0, :, :]
s[1, :, :] = abs(c) * m[1, :, :]
s[2, :, :] = abs(c) * m[2, :, :]
s[3, :, :] = abs(c) * m[3, :, :]
s[4, :, :] = abs(c) * m[4, :, :]
s[5, :, :] = abs(c) * m[5, :, :]

# 4a w loss function
def loss(w, s, c):
    acc = np.zeros([386, 386])
    for k in range(w.shape[0]):
        acc += s[k, :, :] * w[k]
    return norm(acc - c)**2

# 4b w constraint
def sum_to_1(x):
    return np.sum(x) - 1

cons = ({'type': 'eq', 'fun': sum_to_1})

# 4c minimize
x0 = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
bnds = ((0, None), (0, None), (0, None), (0, None), (0, None), (0, None))
res = minimize(loss, x0, args=(s, c), method='SLSQP', bounds=bnds, constraints=cons)
print(res.x)


# w optimization

#1 prep x_k, k=1...5, mask for x and gnd
# x_k = feature space k
x1 = np.asarray(X_w)
x2 = np.asarray(doc_l_tags_3)
x3 = np.asarray(doc_l_mashup_3)
x4 = X_t
x5 = np.asarray(doc_l_providers_3)
gnd = np.asarray(doc_l_tid)

#2 calc c based on gnd (c is a similarity matrix calculated with 2 loop (pair i,j)
c = np.zeros([386, 386])
for i in range(gnd.shape[0]):
    for j in range(gnd.shape[0]):
        if gnd[i] == gnd[j]:
            c[i, j] = 1
        else:
            c[i, j] = -1

#3 for each k, calc s_k, phi2_k
phi2 = np.zeros([5, 386, 386])
s = np.zeros([5, 386, 386])
phi = PCA(n_components=100).fit_transform(x1)
phi2[0, :, :] = phi.dot(phi.T)
s[0, :, :] = abs(c) * phi2[0, :, :]
phi = PCA(n_components=100).fit_transform(x2)
phi2[1, :, :] = phi.dot(phi.T)
s[1, :, :] = abs(c) * phi2[1, :, :]
phi = PCA(n_components=100).fit_transform(x3)
phi2[2, :, :] = phi.dot(phi.T)
s[2, :, :] = abs(c) * phi2[2, :, :]
phi = PCA(n_components=100).fit_transform(x4)
phi2[3, :, :] = phi.dot(phi.T)
s[3, :, :] = abs(c) * phi2[3, :, :]
phi = PCA(n_components=100).fit_transform(x5)
phi2[4, :, :] = phi.dot(phi.T)
s[4, :, :] = abs(c) * phi2[4, :, :]

# 4a w loss function
def loss(w, s, c):
    acc = np.zeros([386, 386])
    # print(w.shape[0]) #
    for k in range(w.shape[0]):
        acc += s[k, :, :] * w[k]
    return norm(acc - c)**2

# 4b w constraint
def sum_to_1(x):
    return np.sum(x) - 1

cons = ({'type': 'eq', 'fun': sum_to_1})

# 4c minimize
x0 = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
bnds = ((0, None), (0, None), (0, None), (0, None), (0, None), (0, None))
res = minimize(loss, x0, args=(s, c), method='SLSQP', bounds=bnds, constraints=cons)
print(res.x)
