# Prediction of cell types in mouse heart tissue from single cell RNA sequencing data with two different machine learning models: Random Forest and Multilayer Perceptron (MLP) neural network

This repository contains the code used for the final project of the Advanced Machine Learning course of the Masters of Bioinformatics at Bologna University.

Authors: Laia Torres Madsdeu, Giacomo Orsini

# Index
- Introduction
  - Aim of the project 
  - The Data
    - Single-cell sequencing
    - Tabula Muris
  - Application of machine learning models
    - Random Forests
    - Multilayer Perceptrons
  - Libraries and modules   
- Data preparation
  - Data retrieval
  - Data preprocessing
    - Remove unclassified cells
    - Remove genes with 0 expression
    - Remove house keeping genes
    - Data normalization
  - Feature selection
    - Variance thresholding
    - Correlation based feature selection
    - Standard scaling
  - Dimensionality reduction
    - PCA
    - t-SNE
    - Umap
  - Feature encoding and dataset splitting
- Models benchmarking
  - Random forest
    - Baseline assessment
    - Hyperparameter tuning
  - Multilayer perceptron
    - Baseline assessment
    - Hyperparameter tuning
- Conclusion

# Introduction 
The project hereby presented was made for the final exam of the Advanced Machine Learning course. The goal was to implement machine learning models in biological tasks. The aims were met, and the project, stored in `AML_LTM_GO.ipynb`, has been positively valued with a grade of 30L. In this READ.ME, some of the instructions and the logic behind the steps are reported, while all the scripts can be found in the `AML_LTM_GO.ipynb`.

- Le H, Peng B, Uy J, Carrillo D, Zhang Y, Aevermann BD, et al. (2022) Machine learning for cell type classification from single nucleus RNA sequencing data. PLoS ONE 17(9): e0275070. https://doi.org/10.1371/journal.pone.0275070
- The Tabula Muris Consortium. Overall coordination. Logistical coordination. et al. Single-cell transcriptomics of 20 mouse organs creates a Tabula Muris. Nature 562, 367–372 (2018). https://doi.org/10.1038/s41586-018-0590-4
- Advanced Machine Learning course notes by Professor Daniele Bonacorsi
- Keras, Sci-kit learn and Tensorflow information sheets.

# Aim of the project
In this project, the performance of two different machine learning methods was evaluated on the same biological task: recognizing cell types in RNA single-cell sequencing data. The machine learning models we used have been Random Forests and Multilayer Perceptrons (MLP).

The cell type recognition task has been carried out on the heart tissue of Mus Musculus (mouse); the cells that compose this tissue are fibroblast, endothelial cell, leukocyte, myofibroblast cell, endocardial cell, cardiac muscle cell, and smooth muscle cell. Moreover, the data used all come from FACS-based full-length transcript analysis.

After preprocessing, different dimensionality reduction/visualization methods (PCA, t-SNE and UMAP) were used, resulting in a total of 4 datasets, which have been used to train the models. For these 4 datasets, both the Random Forest and the MLP have been run with a cross-validation approach to tune different hyperparameters and select the best ones. The best models from the two machine learning methods have been compared.

# The Data
## Single-cell sequencing
Single-cell sequencing is a state-of-the-art biotechnology technique that enables the isolation and sequencing of genetic material from individual cells, with a specific focus on gene expression levels. This allows for the identification and characterization of distinct cell populations within a complex tissue, a task especially valuable in fields such as cancer research, neurobiology, and immunology.

Briefly, the technique begins by isolating individual cells from a tissue sample, often using microfluidics or other specialized methods to separate them. Each cell is then lysed to release its genetic material. To analyze gene expression levels, the mRNA is converted to cDNA, which is then amplified, sequenced, and analyzed. Advanced bioinformatics software then processes and interprets the large datasets.

Thanks to the **Tabula Muris** project, the RNA single-cell sequencing data used in this project has already been cleaned and organized into a table format. Rows represent genes, and columns correspond to codes associated with individual cells. Each cell in the table indicates the expression level of a specific gene.

## Tabula Muris
Tabula Muris is a compendium of single-cell transcriptome data from the model organism Mus musculus (mouse), containing nearly 100,000 cells from 20 organs and tissues. The data allow for a comparison of gene expression in cell types of different tissues and for a comparison of two distinct technical approaches: microfluidic droplet-based 3’-end counting and FACS-based full-length transcript analysis.

The tabula muris data can be accessed through the web page: https://tabula-muris.ds.czbiohub.org/

# Applications of machine learning models
Briefly, machine learning (ML) is a field of artificial intelligence (AI) that enables software systems to learn and improve from experience without being explicitly programmed automatically. It relies on an underlying hypothesis about a model one creates and tries to improve such a model by fitting more and more data into it over time.

The output of this process is a machine learning model: a program created by training an algorithm on data. The model can then take in new data and provide predictions, classifications, or insights based on what it has learned. In particular, ML models are often applied in classification (categorising data into distinct classes) and regression (predicting of continuous numerical values) tasks.

Using machine learning methods on single-cell sequencing data is crucial for uncovering patterns and relationships. Machine learning models can efficiently process high-dimensional data, identify subtle differences and similarities among cells, and classify cell types with high accuracy. This capability is essential for advancing our understanding of cellular functions, uncovering disease mechanisms, and identifying potential therapeutic targets.

In this project, two ML models have been used: Random forests and Neural Networks (Multilayer perceptron)

## Random forests
Random forests (RF) are a powerful and versatile ensemble method used for both classification and regression tasks. Ensemble learning methods are made up of a set of classifiers, where their predictions are aggregated to identify the most popular result. Random forests combine the output of multiple decision trees (DT) to reach a single result. By averaging the predictions of multiple trees, random forests reduce variance and improve generalization to unseen data. They are known for their high accuracy, ease of use, ability to handle large datasets with many features and measure the relative importance of each feature.

## Neural networks: Multilayer perceptron
A multilayer perceptron (MLP) is a type of artificial neural network (NN) that consists of multiple layers of nodes arranged in an input layer, one or more hidden layers, and an output layer. Each node in a layer is connected to every node in the subsequent layer, forming a fully connected network (FCNN). MLPs are designed to learn complex patterns in data through a process called backpropagation, where the network adjusts the weights of connections based on the error of the predictions. This learning process is iterative and aims to minimize the error by optimizing these weights. These types of neural networks are straightforward to implement and are effective for many kinds of classification and regression tasks. Moreover, they are easy to scale and tune.

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
Data preprocessing is a foundational step in machine learning and data science because real-world data is often incomplete, inconsistent, and complex. Preprocessing improves data quality and model performance by transforming raw data into a clean, structured, and analyzable format. This step deals with conditions such as missing data, outliers, and inconsistency.
Four sub-steps typical of data preprocessing of single-cell sequencing data have been carried out: removal of unclassified cells, removal of no-expression genes, removal of housekeeping genes, and data normalisation. The scripts and results of this step can be found in the final report, `AML_LTM_GO.ipynb`.

### Remove unclassified cells
Some cells in the genes-cells file are not annotated in the annotations table because they lack a cell type classification. They have no associated metadata and, therefore, can not be used for the models. 

### Remove genes with no expression in any of the cells
Some genes are not expressed in any of the cells, they have a 0 value in all the columns, therefore they are not informative nor useful for the classification problem.

### Removal of housekeeping genes
Housekeeping genes are genes that are expressed in all cells and cell types; their expression level does not vary between cell types, so they are not informative and unuseful for classification. To determine these genes, the Median Gene Expression within each Cell Type (MGECT) was computed, and genes with zero variance for their MGECT across all cell types were excluded (_Le H, 2022_).

### Data normalisation
RNA sequencing data often exhibit a wide range of expression values with many low counts and a few high counts. This distribution is typically right-skewed. Log transformation helps normalize these values, making the data more symmetrical and reducing the effect of outliers. This is important for the next steps as feature selection methods (variance thresholding, correlation-based selection, and statistical tests) work better on data that has a normalized distribution. 

## Feature selection
Feature selection is an important aspect of preparing data for machine learning, as it reduces the dimensionality of the data, improves model performance, and enhances interpretability by selecting the most informative features and eliminating eventual noise. The steps performed here have been: variance thresholding, correlation-based feature selection and standard scaling. The scripts and results of this step can be found in the final report, `AML_LTM_GO.ipynb`.

### Variance thresholding
Some genes, while not expressed equally between cell types, are expressed at similar levels. Therefore, they are not very informative for the classification task. This step aims to remove genes with very low variance (threshold 2.5 _Le H, 2022_) between cell types.

### Correlation-based feature selection
This step aims to remove redundant and highly correlated genes. The information provided by these genes is redundant.

### Standard scaling 
Standard scaling ensures that each gene contributes equally to the analysis by having zero mean and unit variance. It is essential to apply dimensionality reduction tecnhiques such as PCA, t-SNE or UMAP.

## Dimensionality reduction

Dimensionality reduction is a significant step in machine learning, especially when dealing with high-dimensional datasets such as single-cell sequencing data. It aims to reduce the number of features while retaining the essential information and structure of the data. This simplification facilitates visualization, reduces computational complexity, and can improve the performance of machine learning models by eliminating noise and redundant features. There are three main dimensionality reduction techniques: PCA, t-SNE and UMAP.
The scripts of this step can be found in the final report, `AML_LTM_GO.ipynb`. 4 datasets have been created: one without any dimensionality reduction and the other three, each with a different technique. These 4 datasets (final_df, final_df_pca, final_df_tsne, final_df_umap) were used to test both machine learning models. As can be seen from the plots, the classes in the datasets are not well balanced, as some of them have very few examples; nonetheless, it would be a biological artefact to remove them. For these classes (in particular smooth muscle cells), a worse result in the classification is expected because of the lack of examples.

### Principal Component Analysis (PCA)
It is a widely used method that transforms the data into a set of orthogonal components, capturing the maximum variance with the fewest number of components.

### t-Distributed Stochastic Neighbor Embedding (t-SNE)
It is a powerful non-linear dimensionality reduction technique, particularly for visualization, as it emphasizes the local structure and maintains the high-dimensional pairwise distances in lower dimensions. It works by minimizing the divergence between two distributions: one representing pairwise similarities in the high-dimensional space and one in the low-dimensional space.

### Uniform Manifold Approximation and Projection (UMAP)
It is a newer non-linear dimensionality reduction method that balances the preservation of local and global data structures, providing more interpretable and faster results than t-SNE. It’s based on manifold learning.

## Feature encoding and dataset splitting
Feature encoding is the process of converting categorical data or other types of non-numeric data into a numerical format. The cell types' labels have to be transformed in order for the model to use them.
The datasets have to be split into training and testing subsets, with a proportion of 80 and 20. The models will be fit on the training data, which have to be representative of the whole dataset; after training, the model will be tested on the testing set, where it will predict the class of the unlabeled cells.
K-fold Cross-validation, a procedure that helps ensure that the model's performance is consistent and not dependent on a particular split of the data, has also been performed: the training set has been divided into smaller subsets; at every iteration, one of the splits will be used to test the model and test the performance of each parameter (in the case of hyperparameter tuning). The scripts and results of this step can be found in the final report, `AML_LTM_GO.ipynb`.


