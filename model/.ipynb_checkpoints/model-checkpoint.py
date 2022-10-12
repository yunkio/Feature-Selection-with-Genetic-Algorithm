import numpy as np
import pandas as pd
import random
import sklearn.preprocessing
import statsmodels.api as sm
from tqdm import tqdm
from utils import make_chromosome, select_parents, crossover, train_chromosome

def genetic_algorithm(X, y, population_size=32, cut_off=0.7, crossover_num=17, mutation_rate=0.03, elitism=1, metric='adjusted_r_squared', early_stopping=50, max_generation=1000):
    stag = 0
    best_score = 0
    history = []
    elite_history = []
    
    for i in tqdm(range(max_generation)):
        if i == 0:
            chromosome_list = make_chromosome(X, population_size, cut_off)
        else:
            chromosome_list = child_list

        score_list = train_chromosome(X, population_size, chromosome_list, metric)
        eval_result, elite = evaluate_chromosome(score_list, chromosome_list, elitism)
        child_list = [chromosome_list[elite[0]], chromosome_list[elite[1]]]

        for i in range(int((population_size/2) - 1)):
            parent_1, parent_2 = select_parents(chromosome_list, eval_result)
            new_genes = crossover(parent_1, parent_2, crossover_num, mutation_rate)
            child_list += new_genes

        history.append(max(score_list)) # r-squared
        elite_history.append(chromosome_list[elite[0]])

        if max(score_list) == best_score:
            stag += 1
        else:
            stag = 0

        best_score = history[-1]

        if stag == early_stopping:
            break
    
    return history, elite_history
