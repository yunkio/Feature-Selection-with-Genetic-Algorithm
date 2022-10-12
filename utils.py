import numpy as np
import pandas as pd
import random
import sklearn.preprocessing
import statsmodels.api as sm
from tqdm import tqdm


def make_chromosome(X, population_size, cut_off):
    chromosome_list = []
    for chromosome in range(population_size):
        feature_random_number = []
        for feature in range(X.shape[1]):    
            if random.uniform(0,1) >= cut_off:
                feature_random_number.append(1)
            else:
                feature_random_number.append(0)
        chromosome_list.append(feature_random_number)
    return chromosome_list

def select_parents(chromosome_list, eval_result):
    selected = np.random.choice(range(len(chromosome_list)), 2, p=eval_result['weight'], replace=False)
    parent_1, parent_2 = chromosome_list[selected[0]], chromosome_list[selected[1]]
    return parent_1, parent_2

def crossover(parent_1, parent_2, crossover_num, mutation_rate):
    # make crossover point
    crossover_point = list(sorted(np.random.choice(range(1, len(parent_1)), crossover_num, replace=False)))
    crossover_point.append(len(parent_1))
    
    # split with crossover point
    parent_1_list = []
    parent_2_list = []
    for i, v in enumerate(crossover_point):
        if i == 0:
            parent_1_list.append(parent_1[:v])
            parent_2_list.append(parent_2[:v])
        else:
            parent_1_list.append(parent_1[crossover_point[i-1]:v])
            parent_2_list.append(parent_2[crossover_point[i-1]:v])
            
    # make new generations
    new_genes = []
    
    # crossover
    child_1, child_2 = [], []
    for j in range(crossover_num+1):
        p = random.uniform(0,1)
        if p <= 0.5 : 
            child_1 += parent_1_list[j]
            child_2 += parent_2_list[j]
        else:
            child_1 += parent_2_list[j]
            child_2 += parent_1_list[j]

    # mutation
    for k in range(len(child_1)):
        if random.uniform(0,1) <= mutation_rate:
            if child_1[k] == 0:
                child_1[k] = 1
            else:
                child_1[k] = 0

        if random.uniform(0,1) <= mutation_rate:
            if child_2[k] == 0:
                child_2[k] = 1
            else:
                child_2[k] = 0

    new_genes.append(child_1)
    new_genes.append(child_2)
        
    return new_genes

def train_chromosome(X, y, population_size, chromosome_list, metric='adjusted_r_squared'):
    score_list = []
    for chromosome in range(population_size):
        indices = [i for i,x in enumerate(chromosome_list[chromosome]) if x == 1]
        X_selected = sm.add_constant(X[:,indices])
        model = sm.OLS(y, X_selected).fit()
        
        if metric == 'adjusted_r_squared': 
            score = model.rsquared_adj
        elif metric == 'rmse':
            score = np.sqrt(model.mse_resid)
        score_list.append(score)
    return score_list

def evaluate_chromosome(score_list, chromosome_list, elitism = 1, metric='adjusted_r_squared'):
    if metric == 'adjusted_r_squared': 
        seq = sorted(score_list, reverse=True) # adjusted_r_squared
        score_for_weight = score_list - min(score_list) # adjusted_r_squared
    elif metric == 'rmse':
        seq = sorted(score_list) # rmse
        score_for_weight = max(score_list) - score_list # rmse
    rank = [seq.index(v) for v in score_list]
    sum_score = sum(score_for_weight)
    weight = [(score / sum_score) for score in score_for_weight]
    feature_num = [sum(i) for i in chromosome_list]
    eval_result = pd.DataFrame([score_list, rank, weight, feature_num]).T
    eval_result.columns = ['score', 'rank', 'weight', 'feature_num']
    eval_result['rank'] = eval_result.loc[:, ['rank']].astype('int')
    eval_result['feature_num'] = eval_result.loc[:, ['feature_num']].astype('int')
    
    if elitism:
        elite = eval_result.drop_duplicates().sort_values(['rank', 'feature_num']).index[:elitism*2]
    else:
        elite = False
        
    return eval_result, list(elite)