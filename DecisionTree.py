import pandas as pd
from sklearn import tree
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier

# Importing the dataset
excel_file =  'Excel.csv'
dataset = pd.read_csv(excel_file)

# Label Encoding
for column in dataset.columns:
    if dataset[column].dtype == type(object):
        le = preprocessing.LabelEncoder()
        dataset[column] = le.fit_transform(dataset[column])
        
# DataSet       
X = dataset.iloc[:, [ 1 , 2 , 3 , 4]].values
y = dataset.iloc[:, 5].values


# Gini : Fitting Decision Tree Classification to the Training set 
classifier = DecisionTreeClassifier(criterion = 'gini', random_state = 0)
classifier.fit(X, y)

# Gini : Exporting tree to Text file
tree.export_graphviz(classifier , 'Gini_Tree.txt' ,feature_names=['Age' ,'Income' ,'Student' , 'Credit Rate'])



# Entropy : Fitting Decision Tree Classification to the Training set
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X, y)

# Entropy : Exporting tree to Text file
tree.export_graphviz(classifier , 'Entropy_Tree.txt' ,feature_names=['Age' ,'Income' ,'Student' , 'Credit Rate'])