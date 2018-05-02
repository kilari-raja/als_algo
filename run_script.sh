#!/bin/bash

#Percentage for Transactions used for training. 1 minus this ratio is used as hidden dataset
split_ratio=0.75

#Percentage of Training users that are used for validation of algorithm
#value of 0.001 corresponds roughly to 80 users and
#it takes on average 1 second per user to process the dataset
valid_ratio=0.001

#start value of K (number of vectors to use for SVD)
start_K=50

#end value of K
end_K=50

#step value of K
step_K=10

#threshold for matching customer recipes with other customers
threshold=0.5

#this parameter is used to control overfitting
regularization=0.1

#how many iterations to run the algorithm for
iterations=100

#name of input file which contains transactions
input_file='customer_recipes_purchase_count.csv'

#name of trianing files
train_file='train.csv'

#alpha value for ALS algorithm
alpha=10
python split_data.py -i $input_file -s $split_ratio -v $valid_ratio



for ((K=$start_K;K <= $end_K;K=K+$step_K))
{
  echo "Processing for value of K = $K"

  python implicit_als.py -i $train_file -K $K -r $regularization -iter $iterations -alpha $alpha -t $threshold

}
