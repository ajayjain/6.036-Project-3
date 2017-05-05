#!/usr/bin/env python3
# Changes to this file will not be used during grading

import glob
import os
import pickle
from operator import itemgetter

import project3 as p3
import utils as utils

import pandas as pd

PROJ_DIR = os.path.abspath(os.path.dirname(__file__))
MODELS_DIR = os.path.join(PROJ_DIR, 'models')

if not os.path.exists(MODELS_DIR):
    os.mkdir(MODELS_DIR)

# -------------------------------------------------------------------------------
# Part 1.1
# -------------------------------------------------------------------------------

# toy_data = pd.read_csv(os.path.join(PROJ_DIR, 'toy_data.csv')).as_matrix()

# print('\nPart 1.1: k-means')

# for k in range(1, 6):
#     results = []
#
#     for iter in range(20):
#         results.append(p3.k_means(toy_data, k))
#
#     cost, mu, cluster_assignments = min(results, key=itemgetter(0))
#
#     print(k, cost)
#     utils.plot_kmeans_clusters(toy_data, k, mu, cluster_assignments)

# -------------------------------------------------------------------------------
# Part 1.2
# -------------------------------------------------------------------------------

# print('\nPart 1.2: GMM for k = 1, 2, 3, 4, 5')
#
# GMM_K_MIN_MAX = (1, 5)
# utils.fit_k(p3.GMM, toy_data, *GMM_K_MIN_MAX,
#             MODELS_DIR, verbose=False, d=toy_data.shape[1])

# -------------------------------------------------------------------------------
# Part 1.3
# -------------------------------------------------------------------------------

# print('\nPart 1.3: GMM test')
# utils.test_em_gmm(toy_data)

# -------------------------------------------------------------------------------
# Part 1.4
# -------------------------------------------------------------------------------

# print('\nPart 1.4: GMM plot')
# snaps = glob.glob(os.path.join(MODELS_DIR, 'gmm_*.pkl'))
# snaps.sort(key=utils.get_k)
# for snap in snaps:
#     with open(snap, 'rb') as f_snap:
#         model = pickle.load(f_snap)
#         utils.plot_em_clusters(toy_data, model)

# -------------------------------------------------------------------------------
# Part 2.4
# -------------------------------------------------------------------------------

# print(utils.load_categories('categories.txt').keys())

utils.test_em_cmm()

# -------------------------------------------------------------------------------
# Part 2.5
# -------------------------------------------------------------------------------

# field_cats = utils.load_categories(os.path.join(PROJ_DIR, 'categories.txt'))
# # data = pd.read_csv(os.path.join(PROJ_DIR, 'census_data.csv.gz'))
# data = pd.read_csv(os.path.join(PROJ_DIR, 'tiny_data.csv'))
# ds = data.apply(pd.Series.nunique)
#
# CMM_K_MIN_MAX = (2, 2)
# utils.fit_k(p3.CMM, data, *CMM_K_MIN_MAX, MODELS_DIR, verbose=False, ds=ds)

# -------------------------------------------------------------------------------
# Part 2.6b
# -------------------------------------------------------------------------------

# snaps = glob.glob(os.path.join(MODELS_DIR, 'cmm_*.pkl'))
# snaps.sort(key=utils.get_k)
# ks, bics, lls = [], [], []
# for snap in snaps:
#     with open(snap, 'rb') as f_snap:
#         model = pickle.load(f_snap)
#     ks.append(utils.get_k(snap))
#     lls.append(model.max_ll)
#     bics.append(model.bic)
# utils.plot_ll_bic(ks, lls, bics)

# -------------------------------------------------------------------------------
# Part 2.7
# -------------------------------------------------------------------------------

# K_SHOW = 5  # best K and then some other K
# with open(os.path.join(MODELS_DIR, 'cmm_k%d.pkl' % K_SHOW), 'rb') as f_model:
#     model = pickle.load(f_model)
# utils.print_census_clusters(model, data.columns, field_cats)
