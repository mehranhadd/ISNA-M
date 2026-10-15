*pandas, numpy* and *scikitlearn* libraries are used to train a random forest classifier on the `Clean_ISNA_Dataset.csv`, which contains *Content*(predictor) and *Category*(label) columns. Scikitlearn's `CountVectorizer` is used to create Bag Of Words from contents and `OneHotEncoder` is applied on 83 unique label values.
Data is divided into training and testing subsets with 0.4 test/train ratio. Classifiers with different `n_estimators` values(10, 25, 50, 100, 500 and 1000) are evaluated to find the best value.
|`n_estimators`|***| precision | recall | f1-score |
|:---:|:---:|:---:|:---:|:---:|
|10| micro avg | 0.80 | 0.09 | 0.17 |
|10| macro avg | 0.08 | 0.03 | 0.04 |
|10| weighted avg | 0.21 | 0.09 | 0.12 |
|10| samples avg |0.09 | 0.09 | 0.09 |
|***|***|***|***|***|
|25 |micro avg|0.74|0.09|0.17|
|25 | macro avg | 0.05|0.03|0.03|
|25 | weighted avg |0.18|0.09|0.11|
|25 | samples avg |0.09 | 0.09 | 0.09 |
|***|***|***|***|***|
| 50| micro avg |0.78|0.07|0.12|
|50| macro avg | 0.05  |    0.02 |0.03|
|50| weighted avg | 0.16|      0.07   |   0.09| 
|50| samples avg |0.07|      0.07|      0.07|
|***|***|***|***|***|
|100 | micro avg |0.71|      0.05|      0.09|
|100| macro avg | 0.04|      0.01|      0.02|
|100| weighted avg | 0.14|      0.05|      0.06|
|100| samples avg |0.05|      0.05|      0.05|
|***|***|***|***|***|
| 500 | micro avg |0.75  |    0.04 |     0.08 |
|500| macro avg |0.03 |     0.01  |    0.02|
|500| weighted avg |0.10|      0.04|      0.05|
|500| samples avg |0.04|      0.04|      0.04|
|***|***|***|***|***|
|1000| micro avg |0.77|      0.05|      0.09|
|1000| macro avg |0.04|      0.02|      0.02|
|1000| weighted avg |0.12|      0.05|      0.06 |
|1000| samples avg |0.05|      0.05|      0.05|

The above table is the summary of the outcome. It shows that the best value of the `n_estimators` is 10 and others overfit the data, which is not surprising due to the small amount of the dataset.

The poor results stem from:

1- **Small size of the dataset**. It only contains 167,118 words. This can be tackled by collecting or synthesising more data.

2- **Sparsity of the labels**. Many categories have appeared less than 5 times and may appear only in training or testing subsets after spliting the data. This can be tackled by:
- Collecting or synthesing more data from underrepresented labels.
- Ignoring labels appeared once and make sure each label's samples are present in both training and testing subsets.

3- **Not robust model**. Other hyperparameters should be evaluated through a systematic grid search to find the best setting. Also, various architectures should be tried to find the best solution.