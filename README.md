# Prediction of cell types in mouse heart tissue from single-cell RNA sequencing data with two different machine learning models: Random Forest and Multilayer Perceptron (MLP) neural network

This repository contains the code used for the final project of the Advanced Machine Learning course of the Masters of Bioinformatics at Bologna University.

Authors: Laia Torres Madsdeu, Giacomo Orsini

# Index
- [Introduction](#Introduction)
  - [Aim of the project](#Aim-of-the-project)
  - [The Data](#The-Data)
    - [Single-cell sequencing](#Single-cell-sequencing)
    - [Tabula Muris](#Tabula-Muris)
  - [Application of machine learning models](#Application-of-machine-learning-models)
    - [Random Forests](#Random-Forests)
    - [Multilayer Perceptrons](#Multilayer-Perceptrons)
  - [Libraries and modules](#Libraries-and-modules)
- [Data preparation](#Data-preparation)
  - [Data retrieval](#Data-retrieval)
  - [Data preprocessing](#Data-preprocessing)
    - [Remove unclassified cells](#Remove-unclassified-cells)
    - [Remove genes with 0 expression](#Remove-genes-with-0-expression)
    - [Remove housekeeping genes](#Remove-housekeeping-genes)
    - [Data normalization](#Data-normalization)
  - [Feature selection](#Feature-selection)
    - [Variance thresholding](#Variance-thresholding)
    - [Correlation based feature selection](#Correlation-based-feature-selection)
    - [Standard scaling](#Standard-scaling)
  - [Dimensionality reduction](#Dimensionality-reduction)
    - [PCA](#PCA)
    - [t-SNE](#t-SNE)
    - [Umap](#Umap)
  - [Feature encoding and dataset splitting](#Feature-encoding-and-dataset-splitting)
- [Models benchmarking](#Models-benchmarking)
  - [Hyperparameters tuning](#Hyperparameters-tuning)
  - [Evaluation metrics](#Evaluation-metrics)
- [Results](#Results)
  - [Random forest](#Random-forest)
    - [Baseline assessment](#Baseline-assessment)
    - [Hyperparameter tuning](#Hyperparameter-tuning)
  - [Multilayer perceptron](#Multilayer-perceptron)
    - [Baseline assessment](#Baseline-assessment-1)
    - [Hyperparameter tuning](#Hyperparameter-tuning-1)
- [Conclusion](#Conclusion)

# Introduction 
The project hereby presented was made for the final exam of the Advanced Machine Learning course. The goal was to implement machine learning models in biological tasks. The aims were met, and the project, stored in `AML_LTM_GO.ipynb`, has been positively valued with a grade of 30L. In this `READ.ME`, some of the instructions and the logic behind the steps are reported, while all the scripts can be found in the `AML_LTM_GO.ipynb`.

- Le H, Peng B, Uy J, Carrillo D, Zhang Y, Aevermann BD, et al. (2022) Machine learning for cell type classification from single nucleus RNA sequencing data. https://doi.org/10.1371/journal.pone.0275070
- The Tabula Muris Consortium. Overall coordination. Logistical coordination. et al. Single-cell transcriptomics of 20 mouse organs creates a Tabula Muris. https://doi.org/10.1038/s41586-018-0590-4
- Advanced Machine Learning course notes by Professor Daniele Bonacorsi.
- Keras, Sci-kit learn and Tensorflow information sheets.

# Aim of the project
In this project, the performance of two different machine learning methods was evaluated on the same biological task: **recognizing cell types from RNA single-cell sequencing data**. The machine learning models we used have been **Random Forests** (RF) and **Multilayer Perceptrons** (MLP).

The cell type recognition task has been carried out on the heart tissue of Mus Musculus (mouse); the cells that compose this tissue are _fibroblast_, _endothelial cell_, _leukocyte_, _myofibroblast cell_, _endocardial cell_, _cardiac muscle cell_, and _smooth muscle cell_. Moreover, the data used all come from FACS-based full-length transcript analysis.

After preprocessing, different dimensionality reduction/visualization methods (**PCA**,**t-SNE** and **UMAP**) were used, resulting in a total of 4 datasets, which have been used to train the models. For these 4 datasets, both the RF and the MLP have been run with a **cross-validation** approach to tune different hyperparameters and select the best ones. The best models from the two machine learning methods have been compared.

# The Data
## Single-cell sequencing
Single-cell sequencing is a state-of-the-art biotechnology technique that enables the isolation and sequencing of genetic material from individual cells, with a specific focus on gene expression levels. This allows for the identification and characterization of distinct cell populations within a complex tissue, a task especially valuable in fields such as cancer research, neurobiology, and immunology.

Briefly, the technique begins by isolating individual cells from a tissue sample, often using microfluidics or other specialized methods to separate them. Each cell is then lysed to release its genetic material. To analyze gene expression levels, the mRNA is converted to cDNA, which is then amplified, sequenced, and analyzed. Advanced bioinformatics software then processes and interprets the large datasets.

Thanks to the **Tabula Muris** project, the RNA single-cell sequencing data used in this project has already been cleaned and organized into a table format. Rows represent genes, and columns correspond to codes associated with individual cells. Each cell in the table indicates the expression level of a specific gene.

![image](https://github.com/user-attachments/assets/92934774-1417-4b41-816f-7cb667f73b0f)


## Tabula Muris
Tabula Muris is a compendium of single-cell transcriptome data from the model organism Mus musculus (mouse), containing nearly 100,000 cells from 20 organs and tissues. The data allow for comparing gene expression in cell types of different tissues and for comparing two distinct technical approaches: microfluidic droplet-based 3’-end counting and FACS-based full-length transcript analysis.

The tabula muris data can be accessed through the web page: https://tabula-muris.ds.czbiohub.org/

# Applications of machine learning models
Briefly, machine learning (ML) is a field of artificial intelligence (AI) that enables software systems to learn and improve from experience without being explicitly programmed automatically. It relies on an underlying hypothesis about a model one creates and tries to improve such a model by fitting more and more data into it over time.

The output of this process is a machine learning model: a program created by training an algorithm on data. The model can then take in new data and provide predictions, classifications, or insights based on what it has learned. In particular, ML models are often applied in classification (categorising data into distinct classes) and regression (predicting continuous numerical values) tasks.

Using machine learning methods on single-cell sequencing data is crucial for uncovering patterns and relationships. Machine learning models can efficiently process high-dimensional data, identify subtle differences and similarities among cells, and classify cell types with high accuracy. This capability is essential for advancing our understanding of cellular functions, uncovering disease mechanisms, and identifying potential therapeutic targets.

In this project, two ML models have been used: Random forests and Neural Networks (Multilayer perceptron)

## Random forests
Random forests (RF) are a powerful and versatile ensemble method used for both classification and regression tasks. Ensemble learning methods are made up of a set of classifiers, where their predictions are aggregated to identify the most popular result. Random forests combine the output of multiple **decision trees** (DT) to reach a single result. By averaging the predictions of multiple trees, random forests reduce variance and improve generalization to unseen data. They are known for their high accuracy, ease of use, ability to handle large datasets with many features and measure the relative importance of each feature.

## Neural networks: Multilayer perceptron
A multilayer perceptron (MLP) is a type of **artificial neural network** (NN) that consists of multiple layers of nodes arranged in an input layer, one or more hidden layers, and an output layer. Each node in a layer is connected to every node in the subsequent layer, forming a fully connected network (FCNN). MLPs are designed to learn complex patterns in data through a process called backpropagation, where the network adjusts the weights of connections based on the error of the predictions. This learning process is iterative and aims to minimize the error by optimizing these weights. These types of neural networks are straightforward to implement and are effective for many kinds of classification and regression tasks. Moreover, they are easy to scale and tune.


# Libraries and modules
Different Python libraries and modules have been used in the project and can be found in the following table:

| Task | Library | module |
|------|---------|--------|
| Data preprocessing and visualization | pandas, numpy, pyplot, seaborn | -, -, -, - |
| Feature selection | sklearn.feature_selection, sklearn.preprocessing | VarianceThreshold,  StandardScaler |
| Dimensionality reduction | sklearn.decomposition, sklearn.manifold, umap | PCA,  TSNE, - |
| Feature encoding | sklearn.model_selection, sklearn.preprocessing | train_test_split, LabelEncoder |
| Random Forest | sklearn.ensemble, sklearn.metrics, sklearn.model_selection  | RandomForestClassifier, (accuracy_score, classification_report, confusion_matrix) , GridSearchCV | 
| Multilayer perceptron | tensorflow.keras.utils, keras.models, keras.layers, tensorflow.keras.optimizers, tensorflow.keras.callbacks, keras_tuner, tensorflow.keras.backend, random, tensorflow  | to_categorical, Sequential, (Dense, Dropout), Adam, EarlyStopping, -, -, -, -|

In particular, three libraries are often used for ML implementations in python:
- Scikit-learn (sklearn): provides simple, efficient tools for data analysis and modelling. 
 Includes a wide range of algorithms for tasks like regression, classification, clustering, and dimensionality reduction. It’s particularly valued for its ease of use and integration with other libraries like NumPy and Pandas.
- TensorFlow is a comprehensive, open-source machine learning platform well-suited for both simple and complex tasks. It provides a variety of tools for deep learning, neural network training, and deployment. Keras is often used as an interface for TensorFlow to simplify the model-building process.
- Keras: a high-level neural networks API that makes building machine learning models simpler and more accessible. It runs on top of low-level libraries like TensorFlow and allows for fast prototyping and easy experimentation. With Keras, users can quickly define and train deep neural networks with user-friendly functions.

# Data preparation
## Data retrieval
Data has been retrieved from the Mus Muluscus database (The Tabula Muris Consortium, 2018). Specifically, two files have been retrieved:
1. Metadata table: a file containing information about the single cells, among which the identification code and the cell type annotation (cell_ontology_class).
2. Genes-cells table: a file containing the single-cell sequencing data, where columns are the cells and rows are genes; in the grid, there are the expression levels for each gene in each cell.
In the final report, `AML_LTM_GO.ipynb`, more details can be found.

## Data preprocessing
**Data preprocessing** is a foundational step in machine learning and data science because real-world data is often incomplete, inconsistent, and complex. Preprocessing improves data quality and model performance by transforming raw data into a clean, structured, and analyzable format. This step deals with conditions such as missing data, outliers, and inconsistency.
Four sub-steps typical of data preprocessing of single-cell sequencing data have been carried out: **removal of unclassified cells, removal of no-expression genes, removal of housekeeping genes, and data normalisation**. The scripts and results of this step can be found in the final report, `AML_LTM_GO.ipynb`.

### Remove unclassified cells
Some cells in the genes-cells file are not annotated in the annotations table because they lack a cell type classification. They have no associated metadata and, therefore, can not be used for the models. 

### Remove genes with no expression in any of the cells
Some genes are not expressed in any of the cells, they have a 0 value in all the columns, therefore they are not informative nor useful for the classification problem.

### Removal of housekeeping genes
Housekeeping genes are genes that are expressed in all cells and cell types; their expression level does not vary between cell types, so they are not informative and unuseful for classification. To determine these genes, the Median Gene Expression within each Cell Type (MGECT) was computed, and genes with zero variance for their MGECT across all cell types were excluded (_Le H, 2022_).

### Data normalisation
RNA sequencing data often exhibit a wide range of expression values with many low counts and a few high counts. This distribution is typically right-skewed. Log transformation helps normalize these values, making the data more symmetrical and reducing the effect of outliers. This is important for the next steps, as feature selection methods (variance thresholding, correlation-based selection, and statistical tests) work better on data that has a normalized distribution. 

## Feature selection
Feature selection is an important aspect of preparing data for machine learning, as it reduces the dimensionality of the data, improves model performance, and enhances interpretability by selecting the most informative features and eliminating eventual noise. The steps performed have been: **variance thresholding, correlation-based feature selection and standard scaling**. The scripts and results of this step can be found in the final report, `AML_LTM_GO.ipynb`.

### Variance thresholding
Some genes, while not expressed equally between cell types, are expressed at similar levels. Therefore, they are not very informative for the classification task. This step aims to remove genes with very low variance (threshold 2.5 _Le H, 2022_) between cell types.

### Correlation-based feature selection
This step aims to remove redundant and highly correlated genes. The information provided by these genes is redundant.

### Standard scaling 
Standard scaling ensures that each gene contributes equally to the analysis by having zero mean and unit variance. It is essential to apply dimensionality reduction techniques such as PCA, t-SNE or UMAP.

## Dimensionality reduction
**Dimensionality reduction** is a significant step in machine learning, especially when dealing with high-dimensional datasets such as single-cell sequencing data. It aims to reduce the number of features while retaining the essential information and structure of the data. This simplification facilitates visualization, reduces computational complexity, and can improve the performance of machine learning models by eliminating noise and redundant features. There are three main dimensionality reduction techniques: **PCA, t-SNE** and **UMAP**.
The scripts of this step can be found in the final report, `AML_LTM_GO.ipynb`. 4 datasets have been created: one without any dimensionality reduction and the other three, each with a different technique. These 4 datasets (final_df, final_df_pca, final_df_tsne, final_df_umap) were used to test both machine learning models. As can be seen from the plots, the classes in the datasets are not well balanced, as some of them have very few examples; nonetheless, it would be a biological artefact to remove them. For these classes (in particular smooth muscle cells), a worse result in the classification is expected because of the lack of examples.

### Principal Component Analysis (PCA)
It is a widely used method that transforms the data into a set of orthogonal components, capturing the maximum variance with the fewest number of components.

![image](https://github.com/user-attachments/assets/04cc7d86-53a7-439f-b4d6-873f22509c45)

### t-Distributed Stochastic Neighbor Embedding (t-SNE)
It is a powerful non-linear dimensionality reduction technique, particularly for visualization, as it emphasizes the local structure and maintains the high-dimensional pairwise distances in lower dimensions. It works by minimizing the divergence between two distributions: one representing pairwise similarities in the high-dimensional space and one in the low-dimensional space.

![image](https://github.com/user-attachments/assets/ded99f19-2f42-4e6a-904a-9f50996eb23b)

### Uniform Manifold Approximation and Projection (UMAP)
It is a newer non-linear dimensionality reduction method that balances the preservation of local and global data structures, providing more interpretable and faster results than t-SNE. It’s based on manifold learning.

![image](https://github.com/user-attachments/assets/ab977499-f8a9-4652-91b9-cbac2b9b594d)

## Feature encoding and dataset splitting
**Feature encoding** is the process of converting categorical data or other types of non-numeric data into a numerical format. The cell types' labels must be transformed for the model to use them.
The datasets have to be split into **training** and **testing subsets** (or training set and testing set), with a proportion of 80 and 20. The models will be fit on the training data, which have to be representative of the whole dataset; after training, the model will be tested on the testing set, where it will predict the class of the unlabeled cells.
**K-fold Cross-validation**, a procedure that helps ensure that the model's performance is consistent and not dependent on a particular split of the data, has also been performed: the training set has been divided into smaller subsets; at every iteration, one of the splits will be used to test the model and test the performance of each parameter (in the case of hyperparameter tuning). The scripts and results of this step can be found in the final report, `AML_LTM_GO.ipynb`.

## Models benchmarking
For each machine-learning technique, Random forest (RF and Multilayer Perceptron (MLP), multiple models were created, and multiple benchmarking was carried out:
1. For each method, 4 models were created, considering the 4 datasets of dimensionality reduction: no reduction, PCA, t-SNE, UMAP.
2. For each method, a baseline assessment of the models with default parameters and an assessment with hyperparameter tuning were made. The parameters were tuned using a grid search method. The best model of the hyperparameter tuning benchmarking was selected and compared with the baseline assessment.
3. Finally, the best models of the two ML methods were compared.

### Hyperparameters tuning
Machine learning models have a set of tunable parameters that can substantially change the analysis's outcome. The models were fine-tuned by performing a Grid Search with Cross-Validation. A grid search is a computational technique in which different established sets of hyperparameters are tested in all their possible combinations. The parameters of each ML method considered in the grid search have been:
- Random Forest
  - `n_estimators`: the number of decision trees in the random forest. Increasing n_estimators generally improves the model's performance until a certain point, when additional trees may not significantly improve accuracy but increase computational cost. In this case, the values used have been 100, 500, 1000, and 1200, exploring a range from a relatively small forest to larger forests, which helps in finding the optimal balance between accuracy and computational efficiency. Default: 100.
  - `max_features`: determines the maximum number of features to consider when looking for the best split at each node. By restring (or not) the number of features, the right max_features can prevent overfitting and improve the diversity among the trees in the forest. The options chosen were 'sqrt' (square root of the number of features), 'log2' (log base 2 of the number of features), and None (consider all features). Default: sqrt.
  - `bootstrap`: indicates whether bootstrap samples are used when building trees. Bootstrap sampling introduces randomness and diversity in each tree, which can improve the forest's overall performance. If False, the whole dataset is used to build each tree. Default: True.
- Multilayer perceptron
  - `Number of Layers`: the number of layers that compose the neural network. This can affect the ability to learn complex patterns. Deeper networks can potentially learn more intricate features but can also lead to overfitting. 
  - `Type of Activation Function`: activation functions (e.g., ReLU, sigmoid, tanh) introduce non-linearity, which is crucial for the network to learn and approximate complex functions. ReLU is usually a common choice for hidden layers; softmax for multi-class classification; sigmoid for binary classification.
  - `Units (Number of Neurons)`: determines the capacity of each layer to learn representations. Too few neurons may limit learning capacity; too many may lead to overfitting.
  - `Dropout Rate`: Dropout helps prevent overfitting by randomly setting a fraction of input units to 0 during training, forcing the network to learn redundant representations.
  - `Learning Rate`: the learning rate controls how much to change the model in response to the estimated error each time the model weights are updated. It is a crucial parameter to tune, and the optimal values can vary widely.

### Evaluation metrics
**Evaluation metrics** are essential for benchmarking the performance of machine learning models, particularly for classification tasks, where it’s crucial to understand not only how often a model is correct (accuracy) but also the nature of its mistakes (precision, recall, F1 score, support). The confusion matrix is a tool for evaluating the performance of a classification model by showing the true and predicted classifications side-by-side. The values of the confusion matrix are used to calculate the metrics above.  For a binary classification task, the confusion matrix has four key elements:

- **True Positives** (TP): Cases where the model correctly predicted the positive class.
- **True Negatives** (TN): Cases where the model correctly predicted the negative class.
- **False Positives** (FP) (Type I Error): Cases where the model incorrectly predicted the positive class (a false alarm).
- **False Negatives** (FN) (Type II Error): Cases where the model incorrectly predicted the negative class (missed positives).

1. **Accuracy**: measures the percentage of correct predictions made by the model out of all predictions (TP + TN / N° of predictions).
2. **Precision**: measures the proportion of true positive predictions out of all positive predictions the model made, indicating how often the model’s positive predictions are actually correct (TP/ TP + FP).
3. **Recall**: measures how well the model captures all actual positive cases in the data (TP / TP + FN).
4. **F1 Score**: combines precision and recall into a single metric by calculating their harmonic mean ( 2 * (Precision * Recall / Precision + Recall) )
5. **Support**: the number of actual occurrences of each class in the dataset.

## Results
Hereby, the results obtained by benchmarking the models, as previously stated, are reported and discussed. The full pipeline and scripts can be found in the `AML_LTM_GO.ipynb`.

### Random forest
#### Baseline assessment
In this step, the model with default parameters was tested on the 4 different datasets (no reduction, PCA, t-SNE, UMAP).
The classification reports indicate that the model obtained by training on the original (no dimensionality reduction) data achieved better results, with an accuracy of **0.986**. The model obtained from the PCA dataset performed the worst, with an accuracy of **0.879**.
The overall decrease in the performance of the datasets that went through dimensionality reduction could be explained by the loss of significant features; as much as these techniques aim to retain the essential information and structure of the data, there is a loss of information that could lead to worse classification of some cell types, like in this case.

![image](https://github.com/user-attachments/assets/41027511-f3d2-435c-820c-641a38454781)

#### Hyperparameters tuning
Overall, all models (except t-SNE) performed better than the baseline model. Therefore, tuning the hyperparameters improved the predictions.
The classification reports indicate again that the best model is the one obtained by training on the original data (no dimensionality reduction), with an accuracy of **0.988**. The model obtained from the PCA dataset performed the worst, with an accuracy of **0.881**.
Regarding the parameters, the grid search identified the best parameters (for all datasets):

- `n_estimators`: 1000.
- `max_features`: sqrt.
- `bootstrap`: False (True in PCA, t-SNE and UMAP).

The number of decision trees results in a relatively large random forest, while the 'sqrt' (square root of the number of features) restricts the number of features looked at at the split of each node. The absence of Bootstrap indicates that the whole dataset is used. On a side note, more trees equal more computational complexity and, therefore, more time to be calculated. This has to be kept in mind when tuning this parameter: it is possible that for certain types of tasks and situations, a slightly worse classifier that can be trained in a matter of seconds is better suited than a good classifier that takes hours to be trained. Moreover, the misclassifications between the models are very small (sometimes in the order of 2-3).
The decrease in performance in the datasets that underwent dimensionality reduction can again be explained by the loss of significant features; overall, it seems that applying a dimensionality reduction technique to this kind of dataset, a small, feature-rich single-cell sequencing dataset, could lower the performances when using a Random Forest classifier.
It is also worth mentioning that t-SNE's performance after the hyperparameter tuning was slightly worse than the baseline (0.973 vs 0.972). This can be explained by the fact that, for choosing the parameters, the cross-validation method is performed on the train set, and so, the selected model performed better than the baseline for the training data. However, when applied to the training, it was a bit worse in predicting it.

![image](https://github.com/user-attachments/assets/ab0e1bd9-3b74-4eff-8ef2-165597737eea)

### Multilayer Perceptron
#### Baseline Assessment
The model obtained by training on the original (no dimensionality reduction) data achieved better results, with an accuracy of **0.9896**. The model obtained from the PCA dataset was the one that performed the worst, with an accuracy of **0.8548**.
Indeed, the drop in the loss curves is more sudden in the model trained on the original dataset, and the accuracy also reaches a high plateau very fast.
Overall, the training and validation accuracy curves, as well as the training and validation loss curves,  indicate that the MLP is performing very well, with some differences between the 4 different models.
The overall decrease in the performance of the datasets that went through dimensionality reduction could be explained by the loss of significant features; as much as these techniques aim to retain the essential information and structure of the data, there is a loss of information that could lead to worse classification of some cell types, such as in this case.

![image](https://github.com/user-attachments/assets/b910a82d-d137-4565-881f-848d86480232)

#### Hyperparameters tuning 
Overall, all models performed better than the original model. Therefore, tuning the hyperparameters improved the predictions.

The classification reports follow the pattern already shown with the previous method: the best model is again the one obtained on the original data (no dimensionality reduction), with an accuracy of **0.993**. The model obtained from the PCA dataset performed the worst, with an accuracy of **0.876**.

Regarding the parameters, the grid search has identified the best parameters:

- `Dropout Rate`: 0.3.
- `Activation Function`: relu.
- `Learning Rate`: 0.003373201432675782.
- `Number of Layers`: 1.
- `Units in First Layer`: 64.

The best multilayer perceptron has just one hidden layer, and the first layer is composed of 64 neurons. This architecture is relatively simple for an NN. This means that the data we provided was not particularly complex; this is also coherent when considering that dimensionality reduction lowered the performances, similar to what happened using a Random Forest.

![image](https://github.com/user-attachments/assets/c2447f08-7dae-4e4c-893e-15e7cf651232)

## Conclusions
In conclusion, two famous machine learning methods (Random forest and multilayer Perceptron) have been tested on a dataset of single-cell RNA sequencing data. The computational challenge presented here also represents a significant biological effort, as single-cell sequencing data offer great insight into many biological questions. The choice of using a public, well-curated and broad dataset built on the Mus Musculus (mouse) model organism granted an extensive and representative amount of data. Moreover, the choice of focusing on the heart tissue and building classifiers for the cell types in it defined the borders of our computational challenge.

After a thorough data preprocessing step, we decided to create 4 datasets to evaluate the impact of dimensionality reduction techniques:

- original dataset (no dimensionality reduction)
- PCA dataset
- t-SNE dataset
- UMAP dataset
Using a grid search, together with a cross-validation approach, the best hyperparameters of the models have been found. The hyperparameters that have been tuned are `n_estimators, max_features, bootstrap` for the random forest and `n_layers, dropout_rate, learning_rate, activation_function and n_units` for the MLP.

After training and building our Random forests and Multilayer perceptrons, the result indicates that:

The best random forest obtained an accuracy of **0.988** on the original dataset with the parameters:
- `n_estimators`: 1000
- `max_features`: sqrt
- `Bootstrap`: False
The best multilayer perceptron obtained an accuracy of **0.993** on the original dataset with the parameters:
- `Dropout Rate`: 0.3
- `Activation Function`: relu
- `Learning Rate`: 0.003373201432675782
- `Number of Layers`: 1
- `Units in First Layer`: 64
From these results, the following conclusions can be drawn:

**Multilayer perceptron is a better classifier**: for this task, the Multilayer perceptron has been found to yield better results than the Random Forest. Neural networks are known for their ability to learn highly complex and non-linear relationships in data. Single-cell sequencing datasets often contain intricate patterns and interactions between genes or cells. Random forests, while robust in handling high-dimensional data, may not always extract the most informative features effectively; moreover, random forests may struggle with high-dimensional data if the number of samples is limited, such as in the single-cell dataset, which experiences class unbalance. Neural networks also come with more tuneable parameters, which results in better-tuning potential. The small size of the best performing Neural network indicates that the task is relatively easy and doesn't require much computational complexity (deeper architecture). It is, although quite remarkable, that such a small architecture performs so well on the data, highlighting the power of such models. To be fair, though, it is worth mentioning that both RF and MLP performed very well.

**Dimensionality reduction is not necessary**: the dimensionality reduction techniques applied (PCA, UMAP, t-SNE) have proven to be ineffective in increasing the performances of our classifiers. PCA, in particular, leads to visibly worse results. The explanation for this can be found in the fact that during the data preprocessing and feature selection procedures, we extracted an informative and noise-deprived set of features. Reducing its size led to the loss of significant features rather than a simplification of the complexity, which in turn impacted the classifier's performance. This also proves that the data preprocessing step was done correctly.
