### 引言
在这个Notebook中，我们将展示演化森林的强大和灵活性。演化森林是一种机器学习算法，它将决策树和遗传编程的优点结合在一起，同时支持 Pandas DataFrame 作为输入。这使得使用复杂的数据结构和高效地进行特征构建变得容易。我们将通过一个简单的例子展示如何生成数据集、训练模型、评估性能，并最重要的是，展示如何使用 Pandas DataFrame 格式数据合成高阶特征的过程。

### 步骤 1：导入所需的库

首先，我们导入所有必要的库，包括 Pandas、NumPy 和来自 evolutionary_forest 包的 EvolutionaryForestRegressor。


```python
import sys
sys.path.insert(0, '../')

import random
import string
import pandas as pd
import numpy as np
from sklearn.datasets import make_friedman1
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from evolutionary_forest.forest import EvolutionaryForestRegressor
```

### 步骤 2：生成数据集

我们使用 scikit-learn 库中的 make_friedman1 函数生成一个合成数据集，该函数创建一个包含 100 个样本、五个特征和一个目标变量的数据集。然后，我们将 NumPy 数组转换为 Pandas DataFrame，以便更好地进行解释。


```python
random.seed(0)
np.random.seed(0)

# Generate dataset
X, y = make_friedman1(n_samples=100, n_features=5, random_state=0)

# Convert numpy arrays to pandas dataframe
X = pd.DataFrame(X, columns=list(string.ascii_uppercase[:X.shape[1]]))
y = pd.DataFrame(y, columns=['Target'])
```

### 步骤 3：拆分数据集

为了评估我们的模型性能，我们使用 scikit-learn 库中的 train_test_split 函数将数据集分为训练集和测试集，其中 80% 的数据用于训练，剩余的 20% 用于测试。


```python
# Split dataset
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

### 步骤 4：训练演化森林模型

我们手动设置一些超参数来实例化一个 EvolutionaryForestRegressor 模型，例如最大树高、标准化、选择方法和基本学习器等。然后，我们使用 fit 方法将模型拟合训练数据。


```python
# Train Evolutionary Forest
r = EvolutionaryForestRegressor(max_height=5, normalize=True, select='AutomaticLexicase',
                                gene_num=10, boost_size=100, n_gen=20, n_pop=200, cross_pb=1,
                                base_learner='Random-DT', verbose=True, n_process=1)
r.fit(x_train, y_train)
```

### 第五步：评估模型

为了评估我们训练好的模型的性能，我们使用 scikit-learn 库中的 r2_score 指标，该指标度量了目标变量中可以从输入特征预测的方差比例。R² 分数越接近 1，模型在预测目标变量方面的表现就越好。


```python
# 0.7373017164628979
print(r2_score(y_test, r.predict(x_test)))
```

### 步骤6：分析特征重要性

使用 evolutionary_forest.utils 模块中的 get_feature_importance 和 plot_feature_importance 函数，我们可以可视化数据集中每个特征的重要性。这可以提供有价值的见解，了解哪些特征对模型的预测最为重要。


```python
from evolutionary_forest.utils import get_feature_importance, plot_feature_importance

code_importance_dict = get_feature_importance(r)
plot_feature_importance(code_importance_dict)
```

### 步骤7：合成新特征

在本节中，我们突出展示了演化森林合成新特征的能力，同时保持输出格式为 Pandas DataFrame。这对于跟踪新创建的高阶特征及其与原始特征之间的关系非常有用。

首先，我们使用 feature_append 函数基于之前识别出的顶级特征生成新特征，并将新特征附加到原始 DataFrame 中。输出是一个 Pandas DataFrame，其中列名表示高阶特征，清晰地表示它们与初始特征的关系。这使得我们易于解释和理解新生成的特征。

通过保持输出为 Pandas DataFrame 并提供有意义的列名，演化森林算法确保与我们现有的数据处理流程无缝集成，并允许我们使用熟悉和强大的 Pandas 库进一步分析和操作合成的特征。


```python
from evolutionary_forest.utils import feature_append

# Synthesize new features
code_importance_dict = get_feature_importance(r, simple_version=False)
top_features = list(code_importance_dict.keys())[:len(code_importance_dict) // 2]
new_train = feature_append(r, pd.DataFrame(r.x_scaler.transform(x_train),columns=x_train.columns), top_features, only_new_features=False)
new_test = feature_append(r, pd.DataFrame(r.x_scaler.transform(x_train),columns=x_train.columns), top_features, only_new_features=False)
```


### 结论
在本Notebook中，我们演示了如何使用 Pandas DataFrames 与 Evolutionary Forest Regressor 结合，使其成为处理复杂数据结构的强大工具。通过展示其生成特征重要性和合成新特征的能力，我们希望Evolutionary Forest在更多的数据科学项目中得到应用，并生成更多的洞见。

