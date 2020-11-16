"""
Train a support vector machine on the given dataset with given configuration
"""
import argparse
import math
import re
import sys
import traceback
import pdb
import numpy as np
import tensorflow as tf
import time
from data_utils import *
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.python import debug as tf_debug
from train_utils import *
from sklearn.svm import SVR
import math


parser = argparse.ArgumentParser(description='run ml regressors on dataset',argument_default=argparse.SUPPRESS)
parser.add_argument('--train_data_path', help='path to the training dataset',default=None, type=str, required=False)
parser.add_argument('--test_data_path', help='path to the test dataset', default=None, type=str,required=False)
parser.add_argument('--label', help='output variable', default=None, type=str,required=False)
parser.add_argument('--input', help='input attributes set', default=None, type=str, required=False)
parser.add_argument('--config_file', help='configuration file path', default=None, type=str, required=False)
parser.add_argument('--test_metric', help='test_metric to use', default=None, type=str, required=False)

parser.add_argument('--priority', help='priority of this job', default=0, type=int, required=False)

args,_ = parser.parse_known_args()

hyper_params = {'batch_size':32, 'num_epochs':4000, 'EVAL_FREQUENCY':1000, 'learning_rate':1e-4, 'momentum':0.9, 'lr_drop_rate':0.5, 'epoch_step':500, 'nesterov':True, 'reg_W':0., 'optimizer':'Adam', 'reg_type':'L2', 'activation':'relu', 'patience':100}

# NN architecture

SEED = 66478

def run_regressors(train_X, train_y, valid_X, valid_y, test_X, test_y, kernel, percent, logger=None, config=None):


    def error_rate(predictions, labels, step=0, dataset_partition=''):

        return np.mean(np.absolute(predictions - labels))

    def error_rate_classification(predictions, labels, step=0, dataset_partition=''):
        return 100.0 - (100.0 * np.sum(np.argmax(predictions, 1) == labels) / predictions.shape[0])

    n_train = math.floor(len(train_X)*percent)

    train_X = train_X.reshape(train_X.shape[0], -1).astype("float32")
    valid_X = valid_X.reshape(valid_X.shape[0], -1).astype("float32")
    test_X = test_X.reshape(test_X.shape[0], -1).astype("float32")

    train_y = train_y.reshape(train_y.shape[0]).astype("float32")
    valid_y = valid_y.reshape(valid_y.shape[0]).astype("float32")
    test_y = test_y.reshape(test_y.shape[0]).astype("float32")

    train_data = train_X[0:n_train,:]
    train_labels = train_y[0:n_train]
    test_data = test_X[0:n_train,:]
    test_labels = test_y[0:n_train]
    validation_data = valid_X[0:n_train,:]
    validation_labels = valid_y[0:n_train]

   

    num_input = train_X.shape[1]


    print("train matrix shape of train_X: ",train_X.shape, ' train_y: ', train_y.shape)
    print("valid matrix shape of train_X: ",valid_X.shape, ' valid_y: ', valid_y.shape)
    print("test matrix shape of valid_X:  ",test_X.shape, ' test_y: ', test_y.shape)
    


    # #############################################################################
# Fit regression model
    if kernel == 'rbf':
        svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1,verbose=True)
    elif kernel == 'linear':    
        svr = SVR(kernel='linear', C=100, gamma='auto',verbose=True)
    else:
        svr = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1,
               coef0=1,verbose=True)
    
    print("Train Process")
    svr.fit(train_data, train_labels)
    val_predictions = svr.predict(test_data)
    #test_error = test_metric(val_predictions, test_labels)
    print("Computing score for testing data")
    best_val_error = svr.score(test_data, test_labels)
    print(best_val_error)
    return best_val_error


if __name__=='__main__':
    sparkSession = SparkSession.builder.appName("elemnet_svm").getOrCreate()
    sparkcont = SparkContext.getOrCreate(SparkConf().setAppName("elemnet_svm"))

    args = parser.parse_args()
    config = {}
    config['train_data_path'] = args.train_data_path
    config['test_data_path'] = args.test_data_path
    config['label'] = args.label
    config['input_type'] = args.input
    config['log_folder'] = 'logs_dl'
    config['log_file'] = 'dl_log_' + get_date_str() + '.log'
    config['test_metric'] = args.test_metric
    config['architecture'] = 'infile'
    if args.config_file:
        config.update(load_config(args.config_file))
    if not os.path.exists(config['log_folder']):
        createDir(config['log_folder'])
    logger = Record_Results(os.path.join(config['log_folder'], config['log_file']))
    logger.fprint('job config: ' + str(config))
    
    kernel = 'linear'
    percent = 0.05
    train_X, train_y, valid_X, valid_y, test_X, test_y = load_csv(config['train_data_path'],
                                                                  test_data_path=config['test_data_path'],
                                                                  input_types=config['input_types'],
                                                                  label=config['label'], logger=logger)
    run_regressors(train_X, train_y, valid_X, valid_y, test_X, test_y, kernel, percent, logger=logger, config=config)
    logger.fprint('done')
