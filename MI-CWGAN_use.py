from data_augment import *
#after session divition
#X is the training data, Y is the label for training data
#the size of X is {(101,62,1000), #EEG(trial,channel,times)
#                   (101,24,32), #hbo(trial,channel,times)
#                   (101,24,32)} #hbr(trial,channel,times)
X_train_generated, Y_train_gen_generated = multi_data_augment(X, Y)

# Using X_train_generated and Y_train_gen_generated as inputs for classification network