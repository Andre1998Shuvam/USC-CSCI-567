**K-Nearest Neighbours for Binary Classification (50 points)[¶**](#KNN)**

**General instructions[¶**](#Instructions)**

- In this task you will implement the **K-Nearest Neighbours** algorithm. We provide the bootstrap code and you are expected to complete the classes and functions.
- Do not import libraries other than those already imported in the original code.
- Please follow the type annotations. You have to make the function’s return values match the required type.
- Only modifications in files {knn.py, utils.py} in the "work" directory will be accepted and graded. All other modifications will be ignored. You can work directly on Vocareum, or download all files from "work", code in your own workspace, and then upload the changes (recommended). 
- Click the Submit button when you are ready to submit your code for auto-grading. Your final grade is determined by your **last** submission. 

**Background[¶**](#background)**

In this task, we will use three different functions to measure the distance of two points $x$, $x' \in \mathbb{R}^n$:

- Euclidean distance: $$d(x, x') = \|x-x'\|\_2 =\sqrt{\sum\_{i=1}^{n}\big(x\_{i} - x'\_{i}\big)^{2}}$$
- Minkowski distance: $$ d(x, x') = \|x-x'\|\_3 = \bigg(\sum\_{i=1}^{n}\big|x\_{i} - x'\_{i}\big|^{3}\bigg)^{1/3}$$
- Cosine distance: $$d(x, x') =\begin{cases}1, &\text{if $\|x\|\_2 = 0$ or $\|x'\|\_2=0$} \\ 1-\frac{\langle x, x'\rangle}{\|x\|\_2\|x'\|\_2}. &\text{else} \end{cases}$$

To measure the performance of the algorithm, we will use a widely-used metric called **F1-score** (instead of the "error rate" discussed in the class). You need to self-learn the formula of F1-score from [wikipedia](https://en.wikipedia.org/wiki/F1_score). Note that in this task, a "positive" sample is labeled as "1", and a "negative" one is labeled as "0". 

**Part 1.1 F1 score and distance functions[¶**](#Part-1.1-F1-score-and-Distance-Functions)**

Follow the notes from Background and implement the following items in *util.py*

\- f1\_score

\- class Distance

`    `- euclidean\_distance

`    `- minkowski\_distance

`    `- cosine\_similarity\_distance

**Part 1.2 KNN class[¶**](#Part-1.2-KNN-class)**

Based on what we discussed in the lecture as well as the comments in the code, implement the following items in *knn.py*

\- class KNN

`    `- train

`    `- get\_k\_neighbors

`    `- predict

**Part 1.3 Data transformation[¶**](#Part-1.4-Data-Transformation)**

We are going to add one more step (data transformation) in the data processing part and see how it works. Sometimes, normalization plays an important role to make a machine learning model work. This link might be helpful <https://en.wikipedia.org/wiki/Feature_scaling>. Here, we take two different data transformation approaches.

**Normalizing the feature vector[¶**](#Normalizing-the-feature-vector)**

This one is simple but sometimes may work well. Given a feature vector x, the normalized feature vector is given by $ x' = \frac{x}{\|x\|\_2} $. If a vector is an all-zero vector, we let the normalized vector to be itself.

**Min-max scaling for each feature[¶**](#Min-max-scaling-the-feature-matrix)**

The above normalization is independent of the rest of the data. On the other hand, **min-max normalization** scales each sample in a way that depends on the rest of the data. More specifically, for each feature, we normalize it linearly so that its value are between 0 and 1 across all samples, and in addition, the largest value becomes exactly 1 while the smallest becomes exactly 0. For more information and examples, read the comments in the code. 

You need to implement the following items in *util.py*:

\- class NormalizationScaler

`    `- \_\_call\_\_

\- class MinMaxScaler	

`    `- \_\_call\_\_



**Part 1.4 Hyperparameter tuning[¶**](#Part-1.3-Hyperparameter-Tuning)**

Hyperparameter tuning is an important step for building machine learning models. Here, we can treat the value of k, the distance function, and the data transformation schemes as the hyperparameters of the KNN algorithm. You need to try different combinations of these parameters and find the best model with the highest F1 score. Following the concrete comments in the code and implement the following functions in *util.py*:

\- class HyperparameterTuner

`    `- tuning\_without\_scaling

`    `- tuning\_with\_scaling



**Part 1.5 Testing with *test.py[*¶**](#Use-of-test.py-file)***

There is nothing to implement in this part, but you can make use of the *test.py* file to debug your code and make sure that your implementation is correct. After you have completed all the classes and functions mentioned above, test.py file should run smoothly and show the following if your implementation is correct:

You can also uncomment Line 16 of *data.py*: np.random.shuffle(white), to shuffle the data and further test your code. 

**Grading guideline (50 points)[¶**](#Grading-Guideline-for-KNN-\(50-points\))**

- F-1 score and distance functions - 15 points
- MinMaxScaler and NormalizationScaler - 10 points (5 each)
- Finding best parameters without scaling - 10 points
- Finding best parameters with scaling - 10 points
- Prediction of the best model - 5 points
