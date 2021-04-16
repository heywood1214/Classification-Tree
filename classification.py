import pandas as pd #import library
import numpy as np#import library
from sklearn import datasets #import datasets
from sklearn.datasets import load_breast_cancer #import breast cancer dataset

cancer = load_breast_cancer() #load variable cancer to represent the object

df = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns = np.append(cancer['feature_names'], ['target'])) #create dataframe
df['target']=pd.Series(data.target) #define column "target"
x_var, y_var = cancer.data, cancer.target  #define x and y variable 


df_list = df.values.tolist() #put the whole data frame into list
header = df.columns.tolist() #put all the header into list






from sklearn.model_selection import train_test_split # split data to training and testing
x_train, x_test, y_train, y_test = train_test_split(x_var, y_var, test_size = 0.45, random_state = 0)
x_train_list = x_train.tolist() #create x_training list
x_test_list = x_test.tolist() #create x_testing list
y_train_list = y_train.tolist() #create y_training list
y_test_list = y_test.tolist() #create y_test list

x_train_list = np.array(x_train_list)
y_train_list = np.array(y_train_list)

training_list = np.column_stack((x_train_list,y_train_list))

x_test_list = np.array(x_test_list)
y_test_list = np.array(y_test_list)

testing_list = np.column_stack((x_test_list,y_test_list))

########################################################
def unique_vals(rows, col):  #define a function where I can check for unique values
    return set([row[col] for row in rows]) #use set to filter out all of the unique values



###################################################
def class_counts(rows):  #check the number of class and count them 
    counts = {}  # create an empty dictionary
    for row in rows: # for each row 
        label = row[-1] #our label will be the last row which is target
        if label not in counts: #initialize our label
            counts[label] = 0 #initail label = 0 
        counts[label] += 1 #if the label is already in the class, add 1 to the count
    return counts #return the couint value

###################################################
class Question: #create class called question for decision tree split later
    def __init__(self,column,value): #initalize using a function that requires column and value as input
        self.column = column #the argument requires column
        self.value = value #the arguement requires 

    def match(self,example): #use the column/feature names as the question and use the row as the value for the question
        val = example[self.column] #identify the column 
        return val == self.value #identify the according row using index
    
    def __repr__(self):
        condition = "==" #just to print it nicer 
        return "Is %s %s %s?" % (
            header[self.column], condition, str(self.value)) #to produce the question format : "is question == number"

########################################################

def partition(rows, question): #partition the data set using questions
    true_rows, false_rows = [], [] #create two empties list to match question to see if it is a valid question
    for row in rows: #for each row 
        if question.match(row): #if the question column and value matches
            true_rows.append(row) # then append it to the true_row list
        else:
            false_rows.append(row) #if it doesnt, then append to false rows
    return true_rows, false_rows # exit function by returning the final list for true_rows and false_rows

##############################################################
def gini(rows): #utilize gini index to calculate impurity 
    counts = class_counts(rows)  #count the number of classes again 
    impurity = 1 #initialize impurity = 1 
    for lbl in counts: #for each label/class/target in the 2 classes
        prob_of_lbl = counts[lbl] / float(len(rows)) #the probability of for each label will be the count of the label divided by the # observation in the dataset
        impurity -= prob_of_lbl**2 #gini index = sum of prob_of_lbl(1-prob_of_lbl)
    return impurity #exit function and return the impurity 

##############################################################
def info_gain(left, right, current_uncertainty): #utilizes entropy to get the information,
    p = float(len(left)) / (len(left) + len(right)) #find the weighted sum of entropy for children node
    return current_uncertainty - p * gini(left) - (1 - p) * gini(right) # return entropy(parent) - weighted sum of entropy(children)

##############################################################
def find_best_split(rows): #find the best question to generate the highest information gain through looping each feature 
    best_gain = 0  # initialize the best gain for question = 0 
    best_question = None  # initialize best_question = nothing
    current_uncertainty = gini(rows) #use gini to calcualte the current impurity 
    n_features = len(rows[0]) - 1  # number of columns

    for col in range(n_features):  # loop through each column

        values = set([row[col] for row in rows])  # put all the unique values into a set 

        for val in values:  # loop through each value in the unique value set

            question = Question(col, val) #try to match the question with each value in the unique set

            true_rows, false_rows = partition(rows, question) #try to find the question are valid and partition them to the true and false rows

            if len(true_rows) == 0 or len(false_rows) == 0: #safe guard, if there is none in true_row list or false_row list, we continue
                continue

            gain = info_gain(true_rows, false_rows, current_uncertainty) # find the information gain for each split from the above parition function

            if gain >= best_gain: #if the information gain from the new question is better than using the next value in the set
                best_gain, best_question = gain, question #then replace the best question, and update hte information gain

    return best_gain, best_question #exit function and find the best question and the best gain

##############################################################
class Leaf: #create a leaf node to determine the number of times the particular class appearing in the data to reach that leaf 
    def __init__(self, rows): # requires input of the rows/df_list data
        self.predictions = class_counts(rows) #classify the prediction base on the that classes appear

##############################################################

class Decision_Node:
    def __init__(self, #this decision node keeps track of each branch and each question
                 question, #keep track of the question
                 true_branch, #keep track of the child node 
                 false_branch): #keep track of the child node 
        self.question = question #define the question 
        self.true_branch = true_branch #define the correct branch where it could not split anymore
        self.false_branch = false_branch #define the false branch where it generates could split more 


##############################################################


def build_tree(rows): #input our training data | parition our dataset on each unique value and find the information gain, then keep track of best questions

    gain, question = find_best_split(rows) #find the best question and best gain for each row values and each column feature 

    if gain == 0: #if there is no more information gain 
        return Leaf(rows) #return the leaf which tells us number of times the class appears in the row in the training data that reach this leaf


    true_rows, false_rows = partition(rows, question) #based on the partition from find_best_split, we made sure they are indeed questions by matching columns and values

    true_branch = build_tree(true_rows)#Within the branch itself, we re-run the algorithum to split the correct matched list

    false_branch = build_tree(false_rows)#recursively run the algorithm for eveyrthing else

    return Decision_Node(question, true_branch, false_branch) #return the question both of correct and incorrect list


##############################################################
def print_tree(node, spacing=""): #input a decision node along with the true branch and false branch 

    if isinstance(node, Leaf): # if we hav reached a leaf node
        print (spacing + "Predict", node.predictions)  #return the restults in the leaf node : self.predictions
        return

    print (spacing + str(node.question))#print the question in the node

    print (spacing + '--> True:')#identify the correct branch
    print_tree(node.true_branch, spacing + "  ") #re-run the alogrithm on the correct branch to print more branches

    print (spacing + '--> False:')#identify the wrong branch 
    print_tree(node.false_branch, spacing + "  ") #re-run on the false branch


##############################################################
import time # import library
start_time = time.time() #track current time
my_tree = build_tree(training_list) # 5 mins run time, define my_tree as the tree, I am using 45% for testing because didn't want the run time to be too long
print_tree(my_tree) #show on the terminal
print("--- %s seconds ---" % (time.time() - start_time)) # print the time to run the program


##############################################################
def classify(row, node): # to determine whether if we should follow the true_branch or false_branch


    if isinstance(node, Leaf):#if we alredy reached the leaf
        return node.predictions # then return node prediction

    if node.question.match(row):# if the column value and matches the column 
        return classify(row, node.true_branch) #then re-run it on the true_branch
    else:
        return classify(row, node.false_branch) #if it doesn't match, then go to the false branch


##############################################################

def print_leaf(counts):#just to make the leafs more presentable 
    total = sum(counts.values()) * 1.0 # sum all of counts 
    probs = {} #create an empty dictionary
    for lbl in counts.keys(): #for each label/keys in the dictionary
        probs[lbl] = str(int(counts[lbl] / total * 100)) + "%" #convert it to string, and change the value to a %
    return probs #return the probability


##############################################################
#use testing_data
for row in testing_list: #for each list of data in the testing data 
    print ("Actual: %s. Predicted: %s" % #print the actual label and the predicted class
    (row[-1], print_leaf(classify(row, my_tree))))


testing_sum = 0 #initialize the count of class 1
for row in testing_list: #for each row in the testing_list
    testing_sum = testing_sum + row[-1] #count number of actual class 1
    y_pred_dict = (classify(row, my_tree)) #classify the testing_list
print(y_pred_dict)#predicted number of class 1 
print(testing_sum) #actual number of class 1

accuracy = testing_sum/y_pred_dict[1]#accuracy = number of correct prediction/total number of prediction
print(accuracy) # print accuracy
##############################################################

data = load_breast_cancer()# load data set again

df['target']=pd.Series(data.target) #define column "target"
x_var, y_var = data.data, data.target  #define x and y variable 

start_time = time.time() #track current time


from sklearn.model_selection import train_test_split #import library
x_train, x_test, y_train, y_test = train_test_split(x_var, y_var, test_size = 0.45, random_state = 0) #split data with training 55% and 45% testing data

from sklearn.tree import DecisionTreeClassifier #import library
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0) #use entropy as the decision classifer
classifier.fit(x_train, y_train) #fit the data

y_pred = classifier.predict(x_test) #predict using the classifer

from sklearn.metrics import confusion_matrix #import library for confusion matrix
cm = confusion_matrix(y_test, y_pred) #confusion matrix 

print(cm) #print the confusion matrix
accuracy_2 = (88+154)/(88+9+6+154) 
print(accuracy_2)



print("--- %s seconds ---" % (time.time() - start_time)) # print the time to run the program


#conclusion: therefore the built in algorithm is a lot faster than the custom algorithm, our custom algorithm is 139 seconds, and the built in function is 0.02 second
#The accuracy of our custom function is 82% and the built-in function is 94% 