import os
import glob
import pandas as pd
import numpy as np
from numpy import isnan
import math
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn import metrics
from sklearn.utils import shuffle

# --------------------------------------------class
class global_vars:
    from_adaboost = False
    selected_feature_count = -1
    model_type = -1

global_vars_obj = global_vars()

# ----------------------------------------------functions begin
def print_df_details(mydf):
    print(mydf.columns)
    print(mydf.shape)
    print("-----------")
    print("\n")
#     print(mydf.isna().sum())

# ---------------------------------------read file function section
def readCSV(filename):
    mydf = pd.read_csv(filename)
    
    return mydf

def readCSV_2(filename, splitter, row_count): # only for adult dataset
    header_list = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 
                'hours-per-week', 'native-country', 'label']
    mydf = pd.read_csv(filename, sep=splitter, header = None, engine='python', skiprows=row_count)
    # mydf = pd.read_csv(filename, sep=splitter, header = header_list, engine='python') 
    mydf.columns = header_list

    return mydf

# -----------------------------------preprocessing helper function section
def my_one_hot_encoding(mydf):
    one_hot_col_list = []

    # print(mydf.columns)
    for i in range(len(mydf.columns)):
        col_name = mydf.columns[i]
        if mydf[col_name].dtype == "object" and mydf[col_name].nunique() > 2:
            one_hot_col_list.append(mydf.columns[i])

    # print(one_hot_col_list)

    one_hot_encoded_data = pd.get_dummies(mydf, columns = one_hot_col_list)

    return one_hot_encoded_data

def my_scaling(mydf, scale_col_list): # 4, 7, 8
    scaler = MinMaxScaler()  # StandardScaler() , MinMaxScaler()

    mydf[scale_col_list] = scaler.fit_transform(mydf[scale_col_list])

    return mydf

def my_drop_col(mydf, col_name):
    mydf.drop(col_name, axis = 1, inplace = True)
    mydf = mydf.reset_index(drop = True)

def replace_question_with_mode(mydf): # working_col = [1, 6, 13]
    mydf = mydf.replace(to_replace='?', value=mydf.mode().iloc[0])

    return mydf

# -----------------------------------data split function section
def split_data_2(mydf):
    col_len = len(mydf.columns) - 1
    feature_cols = list(mydf.columns)[:-1]
    label = mydf.columns[col_len]
    
    full_x = mydf[feature_cols] # Features
    full_y = mydf[label] # Target variable
    
    x_train, x_test, y_train, y_test = train_test_split(full_x, full_y, test_size=0.20)

    return x_train, x_test, y_train, y_test


# ---------------------------------------model helper function section
def my_tanh(x):
    e_x = np.exp(x)
    e_minus_x = np.exp(-x)

    res = ((e_x - e_minus_x) / (e_x + e_minus_x))
    return res

def my_loss_func(x, y, h): # lage nai eta
    loss_for_one_data =  (y - h) * (1-np.square(h)) * x
    loss = -np.mean(loss_for_one_data)
    return loss


# -------------------------------------logistic fubction section
def my_LogisticRegression(x_train, y_train, iter_count, alpha):
    data_count = x_train.shape[0] # m
    feature_count = x_train.shape[1] # n

    w = np.zeros([feature_count, 1])

    y_train = y_train.values.reshape(data_count, 1)

    loss_list = []
    
    # print("logit")
    for i in range(iter_count):
        h = my_tanh(np.dot(x_train, w))

        l2_err = my_L2_error(y_train, h)
        if global_vars_obj.from_adaboost == True and l2_err < 0.5:
            # print(i)
            break

        # w = w + alpha * np.dot(x_train.T, (y_train - h) * (1-np.square(h)))
        # w = w + ( alpha * np.dot( x_train.T, (y_train-h) * (1-np.square(h)) ) * (1.0/data_count) )
        
        w = w + np.dot(np.dot(alpha , np.dot(x_train.T , (y_train-h) * (1-np.square(h)))) , (1.0/data_count))

    return w

# ----------------------------------------- sample function section
def get_sample(x_train, y_train, w): # this is basically resample function
    data_count = x_train.shape[0] # m
    feature_count = x_train.shape[1] # n

    index_array = np.arange(data_count)
    sample_index = np.random.choice(index_array, data_count, p=w)

    sample_data = []
    sample_label = []

    for i in range(len(sample_index)):
        index = sample_index[i]
        sample_data.append(x_train.iloc[index])
        sample_label.append(y_train.iloc[index])

    sample_data_df = pd.DataFrame(sample_data)
    sample_label_df = pd.DataFrame(sample_label)

    return sample_data_df, sample_label_df

def get_sample_2(x_train, y_train, w):
    data_count = x_train.shape[0] # m
    feature_count = x_train.shape[1] # n

    frames = [x_train, y_train]
    merged_train = pd.concat(frames, axis = 1)

    # print(merged_train.shape)

    sample_data = merged_train.sample(n = data_count, replace = True, weights = w)

    # print(sample_data.shape)

    feature_cols = list(sample_data.columns)[:-1]
    sample_data_df = sample_data[feature_cols]
    sample_label_df = sample_data[sample_data.columns[-1]]

    return sample_data_df, sample_label_df

# ------------------------------------------adaboost function section
def my_normalize(x):
    return x / np.sum(x)

def my_AdaBoost(x_train, y_train, iter_count, alpha, hypotheses_count):
    x_train = x_train.reset_index(drop = True)
    y_train = y_train.reset_index(drop = True)

    data_count = x_train.shape[0] # m
    feature_count = x_train.shape[1] # n

    fill_value = float(1.0 / data_count)

    w = np.full(data_count, fill_value) # (feature_count, 1)
    
    h = [None] * hypotheses_count # a vector of K hypotheses
    z = [0] * hypotheses_count # a vector of K hypothesis weights
    weighted_majority = []

    for k in range(hypotheses_count):
        sample_data_df, sample_label_df = get_sample_2(x_train, y_train, w)

        # print("k in hypo: " + str(k))
        log_reg_w = my_LogisticRegression(sample_data_df, sample_label_df, iter_count, alpha)
        h[k] = log_reg_w
        
        error = 0.0
        
        for j in range(data_count):
            h_k_xj =  my_logistic_predict_func(x_train.iloc[j], h[k])
            
            if h_k_xj != y_train[j]:
                error = error + w[j]
        if error > 0.5:
            continue
        for j in range(data_count):
            h_k_xj =  my_logistic_predict_func(x_train.iloc[j], h[k])
            if h_k_xj == y_train[j]:
                w[j] = np.dot(w[j], (error / (1-error)))

        w = my_normalize(w)
        # if k == 0:
        #     print(w.shape)

        z[k] = np.log2((1-error) / error)
        # ----------------------------------------------------------

    return h, z

# --------------------------------------------pediction function section
def my_logistic_predict_func(x, w):
    pred_y = my_tanh(np.dot(x, w))
    
    pred_label = []

    for i in range(len(pred_y)):
        if pred_y[i] >= 0:
            pred_label.append(1)
        else:
            pred_label.append(-1)
   
    return pred_label

def my_adaboost_predict_func(x, h, z):
    hypotheses_count = len(z)
    x_into_z_into_h_list = []

    for i in range(hypotheses_count):
        temp_pred = np.dot(z[i], my_tanh(np.dot(x, h[i])))
        x_into_z_into_h_list.append(temp_pred)

    pred_y = np.sum(x_into_z_into_h_list, axis = 0)

    pred_label = []

    for i in range(len(pred_y)):
        if pred_y[i] >= 0:
            pred_label.append(1)
        else:
            pred_label.append(-1)

    return pred_label


# --------------------------------------------metric function section
def my_accuracy(y_test, y_pred):
    acc = np.sum(y_test == y_pred) / len(y_test)

    return acc * 100

def my_L2_error(y_test, y_pred):
    l2 = np.sum(np.power((y_test-y_pred),2)) / len(y_test)
    return l2

def get_metric(y, y_pred):
    acc = metrics.accuracy_score(y, y_pred) * 100

    TN, FP, FN, TP = metrics.confusion_matrix(y, y_pred).ravel()

    accuracy = float((TP + TN) / (TP + TN + FP + FN))
    recall = float(TP / (TP + FN))
    specificity = float(TN / (TN + FP))
    precision = float(TP / (TP + FP))
    false_discovery_rate = float(FP / (FP + TP))
    f1_score = float((2 * precision * recall) / (precision + recall))
    
    cm = metrics.confusion_matrix(y, y_pred)
    # print(cm)

    return cm, accuracy * 100, recall * 100, specificity * 100, precision * 100, false_discovery_rate * 100, f1_score

# ------------------------------------preprocessing function section
def preprocessing_churn(churn_df):
    my_drop_col(churn_df, "customerID")

    churn_df["TotalCharges"] = churn_df["TotalCharges"].replace(to_replace=" ", value=0.0)
    churn_df["TotalCharges"] = churn_df.TotalCharges.astype(np.float64)

    # replacing yes = 1, no = 0
    churn_df["Churn"] = churn_df["Churn"].replace(to_replace="Yes", value=1)
    churn_df["Churn"] = churn_df["Churn"].replace(to_replace="No", value=-1)
    churn_df = churn_df.replace(to_replace="Yes", value=1)
    churn_df = churn_df.replace(to_replace="No", value=0)
    churn_df = churn_df.replace(to_replace="Female", value=0)
    churn_df = churn_df.replace(to_replace="Male", value=1)

    churn_df = my_one_hot_encoding(churn_df)

    # moving label column at the end of the dataframe
    label_df = churn_df.pop('Churn')
    churn_df['Churn'] = label_df

    scale_col_list = ['tenure', 'MonthlyCharges', 'TotalCharges']
    churn_df = my_scaling(churn_df, scale_col_list)
    
    return churn_df


def preprocessing_adult(adult_df):
    adult_df = replace_question_with_mode(adult_df)

    adult_df["label"] = adult_df["label"].replace(to_replace=">50K", value=1)
    adult_df["label"] = adult_df["label"].replace(to_replace="<=50K", value=-1)
    adult_df["sex"] = adult_df["sex"].replace(to_replace="Female", value=1)
    adult_df["sex"] = adult_df["sex"].replace(to_replace="Male", value=0)

    adult_df = my_one_hot_encoding(adult_df) # (32561, 105)

    label_df = adult_df.pop("label")
    adult_df["label"] = label_df

    scale_col_list = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    adult_df = my_scaling(adult_df, scale_col_list)
    
    return adult_df

def preprocessing_creditcard(creditcard_df):
    creditcard_df["Class"] = creditcard_df["Class"].replace(to_replace=0, value=-1)
    scale_col_list = creditcard_df.columns[:-1]
    
    creditcard_df = my_scaling(creditcard_df, scale_col_list)
    
    return creditcard_df


def get_info_gain(mydf):
    col_len = len(mydf.columns) - 1
    feature_cols = list(mydf.columns)[:-1]
    label = mydf.columns[col_len]
    full_x = mydf[feature_cols] # Features
    full_y = mydf[label] # Target variable

    ig_list = mutual_info_classif(full_x, full_y)

    index_ig_list = np.argsort(ig_list)

    ig_list = []
    for i in range(len(index_ig_list)):
        j = len(index_ig_list) -1 - i
        index = index_ig_list[j]
        ig_list.append(index)

    return ig_list  

def get_selected_feature_df(mydf, selected_feature_count):
    # print("in selected features")
    ig_idx_list = get_info_gain(mydf)
    ig_idx_list = ig_idx_list[:selected_feature_count]
    
    ig_col_list = []
    for i in range(len(ig_idx_list)):
        col_idx = ig_idx_list[i]
        ig_col_list.append(mydf.columns[col_idx])

    last_col = len(mydf.columns) - 1
    last_col_name = mydf.columns[last_col]
    ig_col_list.append(last_col_name)

    mydf = mydf.loc[:, ig_col_list]

    return mydf


# ----------------------------------------------main function section
def run_logistic(my_x_train, my_y_train, my_x_test, my_y_test, iter_count, alpha):
    w = my_LogisticRegression(my_x_train, my_y_train, iter_count, alpha)

    my_y_pred_train = my_logistic_predict_func(my_x_train, w)
    cm, accuracy, recall, specificity, precision, false_discovery_rate, f1_score = get_metric(my_y_train, my_y_pred_train)
    print("train accuracy: " + str(accuracy))
    print("train recall: " + str(recall))
    print("train precision: " + str(precision))
    print("train specificity: " + str(specificity))
    print("train false_discovery_rate: " + str(false_discovery_rate))
    print("train f1_score: " + str(f1_score))

    my_y_pred = my_logistic_predict_func(my_x_test, w)
    cm, accuracy, recall, specificity, precision, false_discovery_rate, f1_score = get_metric(my_y_test, my_y_pred)
    print("test accuracy: " + str(accuracy))
    print("test recall: " + str(recall))
    print("test precision: " + str(precision))
    print("test specificity: " + str(specificity))
    print("test false_discovery_rate: " + str(false_discovery_rate))
    print("test f1_score: " + str(f1_score))


def run_adaboost(my_x_train, my_y_train, my_x_test, my_y_test, iter_count, alpha):
    global_vars_obj.from_adaboost = True
    K_list = [5, 10, 15, 20]
    for i in range(len(K_list)):
        hypotheses_count = K_list[i]
        if hypotheses_count > 0:
            print("k: " + str(hypotheses_count))
            h, z = my_AdaBoost(my_x_train, my_y_train, iter_count, alpha, hypotheses_count)

            my_y_pred_train =  my_adaboost_predict_func(my_x_train, h, z)
            cm, accuracy, recall, specificity, precision, false_discovery_rate, f1_score = get_metric(my_y_train, my_y_pred_train)
            print("train accuracy: " + str(accuracy))
            print("train recall: " + str(recall))
            print("train precision: " + str(precision))
            print("train specificity: " + str(specificity))
            print("train false_discovery_rate: " + str(false_discovery_rate))
            print("train f1_score: " + str(f1_score))
            
            my_y_pred = my_adaboost_predict_func(my_x_test, h, z)
            cm, accuracy, recall, specificity, precision, false_discovery_rate, f1_score = get_metric(my_y_test, my_y_pred)
            print("test accuracy: " + str(accuracy))
            print("test recall: " + str(recall))
            print("test precision: " + str(precision))
            print("test specificity: " + str(specificity))
            print("test false_discovery_rate: " + str(false_discovery_rate))
            print("test f1_score: " + str(f1_score))


def subset_data(creditcard_df):
    df_0 = creditcard_df.query("Class == 0")[:][:5000] # 20000
    df_1 = creditcard_df.query("Class == 1")

    frames = [df_0, df_1]
    sub_creditcard_df = pd.concat(frames)

    sub_creditcard_df = shuffle(sub_creditcard_df)

    return sub_creditcard_df

def work_with_dataset(my_df, selected_feature_count, dataset_no, model_type):
    # my_df = preprocessing_churn(my_df)
    if dataset_no == 1:
        my_df = preprocessing_churn(my_df)
    elif dataset_no == 3:
        my_df = preprocessing_creditcard(my_df)

    if model_type == 2:
        my_df = get_selected_feature_df(my_df, selected_feature_count)

    my_df.insert(0, "Ones", 1)

    iter_count = 1000
    alpha = 0.05 # 0. 1-0.5   000002

    my_x_train, my_x_test, my_y_train, my_y_test = split_data_2(my_df) # dataframe

    if model_type == 1:
        run_logistic(my_x_train, my_y_train, my_x_test, my_y_test, iter_count, alpha)

    elif model_type == 2:
        run_adaboost(my_x_train, my_y_train, my_x_test, my_y_test, iter_count, alpha)


def work_with_dataset_2(train_df, test_df, selected_feature_count, model_type):
    train_df = preprocessing_adult(train_df)
    my_drop_col(train_df, "native-country_Holand-Netherlands")

    test_df["label"] = test_df["label"].replace(to_replace=">50K.", value=">50K")
    test_df["label"] = test_df["label"].replace(to_replace="<=50K.", value="<=50K")
    test_df = preprocessing_adult(test_df)

    if model_type == 2:
        train_df = get_selected_feature_df(train_df, selected_feature_count)
        train_col_list = train_df.columns
        test_df = test_df.loc[:, train_col_list]


    train_df.insert(0, "Ones", 1)
    test_df.insert(0, "Ones", 1)

    col_len = len(train_df.columns) - 1
    feature_cols = list(train_df)[:-1]
    label = train_df.columns[col_len]
    
    my_x_train = train_df[feature_cols] # Features
    my_y_train = train_df[label] # Target variable

    col_len = len(test_df.columns) - 1
    feature_cols = list(test_df)[:-1]
    label = test_df.columns[col_len]
   
    my_x_test = test_df[feature_cols] # Features
    my_y_test = test_df[label] # Target variable

    iter_count = 1000
    alpha = 0.05 # 0. 1-0.5   000002

    if model_type == 1:
        run_logistic(my_x_train, my_y_train, my_x_test, my_y_test, iter_count, alpha)
    elif model_type == 2:
        run_adaboost(my_x_train, my_y_train, my_x_test, my_y_test, iter_count, alpha)
    else:
        pass


path = os.getcwd() + "/data/"

dataset_no = int(input("Please enter dataset no: (1 for dataset 1, 2 for datset 2, 3 for dataset 3) "))
model = int(input("Please enter which type of model: (1 to run logistic regression , 2 to run adaboost) "))
global_vars_obj.model_type = model
if model == 2:
    selected_feature_count = int(input("Please enter the number of features to be selected through information gain: "))
    global_vars_obj.selected_feature_count = selected_feature_count

if dataset_no == 1:
    # filename1 = path + "archive/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    filename1 = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    churn_df = readCSV(filename1)
    # print(churn_df.shape)
    if global_vars_obj.model_type == 1:
        global_vars_obj.selected_feature_count = churn_df.shape[1]
    # print(global_vars_obj.model_type)
    # print(global_vars_obj.selected_feature_count)
    work_with_dataset(churn_df, global_vars_obj.selected_feature_count, 1, global_vars_obj.model_type)

elif dataset_no == 2:
    # filename2 = path + "adult/adult.data"
    filename2 = "adult.data"
    adult_train_df = readCSV_2(filename2, ", ", 0)
    # print(adult_train_df.shape)
    # filename2_test = path + "adult/adult.test"
    filename2_test = "adult.test"
    adult_test_df = readCSV_2(filename2_test, ", ", 1)
    # print(adult_test_df.shape)
    if global_vars_obj.model_type == 1:
        global_vars_obj.selected_feature_count = adult_train_df.shape[1]
    # print(global_vars_obj.model_type)
    # print(global_vars_obj.selected_feature_count)
    work_with_dataset_2(adult_train_df, adult_test_df, global_vars_obj.selected_feature_count, global_vars_obj.model_type)

elif dataset_no == 3:
    # filename3 = path + "archive(1)/creditcard.csv"
    filename3 = "creditcard.csv"
    creditcard_df = readCSV(filename3)
    # print(creditcard_df.shape)
    creditcard_df = subset_data(creditcard_df)
    # print(creditcard_df.shape)
    if global_vars_obj.model_type == 1:
        global_vars_obj.selected_feature_count = creditcard_df.shape[1]
    # print(global_vars_obj.model_type)
    # print(global_vars_obj.selected_feature_count)
    work_with_dataset(creditcard_df, global_vars_obj.selected_feature_count, 3, global_vars_obj.model_type)