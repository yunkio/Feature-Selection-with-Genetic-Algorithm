import numpy as np
import pandas as pd
import random
import sklearn.preprocessing
import statsmodels.api as sm
from tqdm import tqdm
from utils import make_chromosome, select_parents, crossover, train_chromosome, evaluate_chromosome

def genetic_algorithm(X, y, population_size=32, cut_off=0.7, crossover_num=5, mutation_rate=0.03, elitism=1, metric='adjusted_r_squared', early_stopping=50, max_generation=1000):

    stag = 0
    best_score = 0
    history = []
    elite_history = []

    for i in tqdm(range(max_generation)):
        if i == 0:
            chromosome_list = make_chromosome(X, population_size, cut_off)
        else:
            chromosome_list = child_list

        score_list = train_chromosome(X, y, population_size, chromosome_list, metric)
        eval_result, elite = evaluate_chromosome(score_list, chromosome_list, elitism, metric)
        child_list = [chromosome_list[elite[i]] for i in range(elitism*2)]

        for i in range(int((population_size/2) - elitism)):
            parent_1, parent_2 = select_parents(chromosome_list, eval_result)
            new_genes = crossover(parent_1, parent_2, crossover_num, mutation_rate)
            child_list += new_genes
        
        if metric == 'adjusted_r_squared':
            history.append(max(score_list)) # r-squared
        elif metric == 'rmse':
            history.append(min(score_list)) # r-squared
        elite_history.append(chromosome_list[elite[0]])
        
        if metric == 'adjusted_r_squared':
            if max(score_list) == best_score:
                stag += 1
            else:
                stag = 0
        elif metric == 'rmse':
            if min(score_list) == best_score:
                stag += 1
            else:
                stag = 0
                
        best_score = history[-1]

        if stag == early_stopping:
            break
    
    return history, elite_history