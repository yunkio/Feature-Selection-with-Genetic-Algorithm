
# 유전 알고리즘을 활용한 변수 선택 알고리즘

> 유전 알고리즘을 활용한 변수 추출 알고리즘 튜토리얼 페이지 입니다.

유전 알고리즘 *Genetic Algorithm*은 생명체의 생식 과정을 모사한 진화 알고리즘의 한 종류입니다. 우수한 유전자를 생식을 통해 다음 세대에 전달하는 것을 목적으로 합니다.

변수 선택에서는, 이러한 유전 알고리즘의 특징을 활용하여 결과에 유의미한 영향을 끼치는 변수를 선택하여 데이터의 차원을 줄이는 것을 목적으로 합니다.

![image](https://user-images.githubusercontent.com/35906602/195421636-5fbd2035-6fd1-40e5-998b-59c2b4b0a06f.png)

변수 추출 알고리즘


1. 염색체 초기화
2. 염색체의 선택 변수별 모델 학습
3. 각 염색체 적합도 평가
4. 우수 염색체 선텍
5. 다음 세대 염색체 생성

변수 추출 알고리즘은 크게 5단계로 이루어집니다. 설정된 종료 조건을 만족할 때까지 위 단계를 반복적으로 수행하여 최적의 변수 set을 찾아내게 됩니다.본 튜토리얼에서는 각 단계별로 구현하고, 각 하이퍼파라미터가 알고리즘에 미치는 영향을 알아보게 됩니다.


## 사용법

```python
from model import genetic_algorithm

history, elite_history = genetic_algorithm(X, y, population_size=32, cut_off=0.7, crossover_num=5,
mutation_rate=0.03, elitism=1, metric='adjusted_r_squared',
early_stopping=50, max_generation=1000)
```

### 하이퍼 파라미터

**popluation_size : *int, default = 32*** 

각 세대를 구성하는 유전자 조합의 수 입니다. 숫자가 클수록 한 세대에 더 많은 유전자 조합을 시도하게 됩니다.

**cut_off : *float, default = 0.7*** 

0 에서 1 사이의 값을 가집니다. 값이 클수록 염색체 초기화시에 더 적은 수의 변수를 선택합니다. 

**crossover_num : *int, default = 5***
 
교배 지점 (Crossover Point)의 수 입니다. 0부터 n(유전자 갯수)-1까지 가능합니다.

**mutation_rate : *float, default = 0.03***

변이율로, 0 에서 1 사이의 값을 가집니다. 각 유전자는 해당 값의 확률로 무작위로 변화하게 됩니다. Local optima에 빠질 위험을 줄이지만, 너무 높으면 수렴이 느려집니다.

**elitism : *int, default = 1***

각 세대별로 가장 뛰어난 유전자쌍을 다음 세대로 넘겨줍니다. 설정한 값의 2배만큼의 염색체를 넘겨주게 됩니다. 0으로 설정하면 유전자쌍을 넘기지 않습니다.

**metric : *{'adjusted_r_squared', 'rmse'}***

염색체를 평가할 때 사용할 평가 지표입니다. rmse 혹은 adjusted r squared 값을 사용 가능합니다.

**early_stopping : *int, default = 50*** 

설정한 값만큼의 세대가 지나도 더 뛰어난 염색체가 등장하지 않는다면 알고리즘을 종료합니다.
 
**max_generation : *int, default = 1000***

최대 세대 수입니다. 최대 세대에 도달하면 멈추게 됩니다.


코드에 대한 자세한 설명은 Tutorial.ipynb 파일을 참고해주세요.
