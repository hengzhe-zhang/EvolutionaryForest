> 演化森林开源算法包（Evolutionary Forest (TEVC 2021), SR-Forest (TEVC
> 2023)）：https://github.com/hengzhe-zhang/EvolutionaryForest

### 引言

本文的目的是演示如何在演化森林回归模型中定义和使用自定义原语。

演化森林是一种机器学习算法，它将遗传编程与决策树相结合，创建针对回归任务的集成模型。它使用遗传编程来演化一组复杂特征，每个特征表示为一个计算机程序。每个程序都包含原语，这些原语表示可用于构建复杂特征的函数或运算符。

默认情况下，演化森林带有一组内置原语，例如加法、减法、乘法和除法。但是，用户可以定义自己的自定义原语来构建更复杂的特征。

代码首先使用默认的原语训练一个演化森林回归模型的实例。然后，我们定义了一个新的自定义原语
Sin，用于计算给定值的正弦值。EvolutionaryForestRegressor 构造函数的 custom_primitive 参数用于传递字典以指定自定义原语。

首先，我们使用默认的原语训练了一个演化森林回归模型的实例。实验结果显示 $R^2$ 得分为0.72。

```python
import sys
sys.path.insert(0, '../')

import random
import numpy as np
from sklearn.datasets import make_friedman1
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from evolutionary_forest.forest import EvolutionaryForestRegressor

random.seed(0)
np.random.seed(0)

# Generate dataset
X, y = make_friedman1(n_samples=100, n_features=5, random_state=0)
# Split dataset
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train Evolutionary Forest
r = EvolutionaryForestRegressor(max_height=5, normalize=True, select='AutomaticLexicase',
                                gene_num=10, boost_size=100, n_gen=20, n_pop=200, cross_pb=1,
                                base_learner='Random-DT', verbose=True, n_process=1)
r.fit(x_train, y_train)
# 0.7252972657876586
print(r2_score(y_test, r.predict(x_test)))
```

### 自定义原语

现在，我们使用 EvolutionaryForestRegressor 构造函数的 custom_primitive 参数来传递一个字典，将每个自定义原语的名称映射到一个包含相应函数和它所需参数数量的元组。

接下来，我们使用自定义原语训练一个新的演化森林回归模型实例，并使用 $R^2$
得分在测试集上评估其性能。实验结果显示性能有所提高，$R^2$ 得分从 0.72 提高到 0.73。这种改进表明，在演化森林回归模型中使用自定义原语可以提高其在回归任务上的性能。

```python
# Define Custom Primitives
r = EvolutionaryForestRegressor(max_height=5, normalize=True, select='AutomaticLexicase',
                                gene_num=10, boost_size=100, n_gen=20, n_pop=200, cross_pb=1,
                                base_learner='Random-DT', verbose=True, n_process=1,
                                basic_primitive='Add,Mul,Div',
                                custom_primitive={
                                    'Sin': (np.sin, 1)
                                })
r.fit(x_train, y_train)
# 0.734064764894771
print(r2_score(y_test, r.predict(x_test)))
```

总的来说，本代码演示了如何通过定义和使用自定义原语来扩展演化森林回归模型的功能。这对于解决更复杂的回归问题非常有用。
