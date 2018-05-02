import sys
import pandas as pd
import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
import random
import argparse
from datetime import datetime

from sklearn.preprocessing import MinMaxScaler

import implicit

def build_model(input_file, K, regularization, iterations, alpha_val):
    #loading customer purchase history from CSV file
    raw_data = pd.read_csv(input_file, index_col=False)
    raw_data.columns = ['user', 'recipe', 'counts']
    #print(len(raw_data))

    # Drop NaN columns
    data = raw_data.dropna()
    data = data.copy()
    #print(len(data))

    # Create a numeric user_id and artist_id column
    data['user'] = data['user'].astype("category")
    data['recipe'] = data['recipe'].astype("category")
    data['user_id'] = data['user'].cat.codes
    data['recipe_id'] = data['recipe'].cat.codes

    #print(data)
    user_lookup = data[['user_id', 'user']].drop_duplicates()
    #item_lookup = data[['recipe_id', 'recipe']].drop_duplicates()
    #item_lookup['recipe_id'] = item_lookup.recipe_id.astype(str)

    #data = data.drop(['user', 'recipe'], axis=1)
    #item_lookup['artist_id'] = item_lookup.artist_id.astype(str)
    # The implicit library expects data as a item-user matrix so we
    # create two matricies, one for fitting the model (item-user)
    # and one for recommendations (user-item)
    sparse_item_user = sparse.csr_matrix((data['counts'].astype(float), (data['recipe_id'], data['user_id'])))
    sparse_user_item = sparse.csr_matrix((data['counts'].astype(float), (data['user_id'], data['recipe_id'])))

    # Initialize the als model and fit it using the sparse item-user matrix
    model = implicit.als.AlternatingLeastSquares(K, regularization, iterations)

    # Calculate the confidence by multiplying it by our alpha value.
    data_conf = (sparse_item_user * alpha_val).astype(np.float32)

    #print(data_conf.dtype)
    #print(data_conf.shape)
    #Fit the model
    model.fit(data_conf)

    return model, sparse_user_item, data, user_lookup
'''

#---------------------
# FIND SIMILAR ITEMS
#---------------------

# Find the 10 most similar to Jay-Z
item_id = 147068 #Jay-Z
n_similar = 10

# Use implicit to get similar items.
similar = model.similar_items(item_id, n_similar)

# Print the names of our most similar artists
for item in similar:
    idx, score = item
    print data.artist.loc[data.artist_id == idx].iloc[0]

'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Running Implicity ALS Algorithm for Recipe Recommendation')
    required_args = parser.add_argument_group('required arguments')

    required_args.add_argument(
        "-i", "--input-file", dest="input_file", required=True, \
        help="Input file with customer purchase data, Currently only .csv files are supported"
    )

    required_args.add_argument(
        "-K", "--K", dest="K", type=int, required=True, \
        help="Number of SVD Vectors"
    )

    required_args.add_argument(
        "-r", "--r", dest="regularization", type=float, required=True, \
        help="Regularization parameter needed to control overfitting"
    )

    required_args.add_argument(
        "-iter", "--iterations", dest="iterations", type=int, required=True, \
        help="Number of iterations to run the ALS algorithm for"
    )

    required_args.add_argument(
        "-alpha", "--alpha", dest="alpha_val", type=int, required=True, \
        help="Alpha value needed for ALS algorithm"
    )

    required_args.add_argument(
        "-t", "--threshold", dest="threshold", type=float, required=True, \
        help="Threshold used for matching customer recipes with other customers"
    )

    arguments = parser.parse_args()

    default_start_time = datetime.now()
    start_time = datetime.now()

    model, sparse_user_item, data, user_lookup = build_model(arguments.input_file, arguments.K, arguments.regularization, arguments.iterations, arguments.alpha_val)



    sum = 0.0
    valid = pd.read_csv('valid.csv', index_col=False)
    index = 1
    count = 1

    for u,row in valid.iterrows():
        print('Predicting for User # {} out of {} users'.format(index, len(valid)))
        index += 1

        #print(row['user_id'])

        #finding the user_id using the username
        user_id = int(data.user_id.loc[data.user == row['user_id']].iloc[0])
        predictions = model.recommend(user_id, sparse_user_item, N = 5)
        #print(predictions)

        predicted_recipes = []
        scores = []
        #extracting all predicted recipes
        for item in predictions:
            idx, score = item
            predicted_recipes.append(data.recipe.loc[data.recipe_id == idx].iloc[0])
            scores.append(score)

        #finding actual purchases for the consumer
        user_df = data[data['user_id'] == user_id]
        actual_recipes = user_df['recipe'].tolist()

        size_actual_repices = float(len(actual_recipes))
        #reading in the validation dataset
        df = pd.read_csv('test.csv', index_col=False)
        #print(validation_set)
        #doing a group by users so that we have customers and their corresponding recipes in one row
        grouped_data = df.groupby('user_id')['recipe_name'].agg(lambda col: ','.join(col)).reset_index()

        matching_recipes = []

        #iterating over each user and getting their recipes
        for k,v in grouped_data.iterrows():
            #print(v['user_id'])
            customer_recipe_list = v['recipe_name'].strip().split(',')
            #print(customer_recipe_list)
            #print(set(actual_recipes).intersection(customer_recipe_list))
            size_match = len(set(actual_recipes).intersection(customer_recipe_list))
            if size_match != 0:
                per_match = size_match / size_actual_repices
                #print('Percentage match: %f' % per_match)
                if per_match >= arguments.threshold:
                    matching_recipes.extend(set(customer_recipe_list))

        matching_recipes = set(matching_recipes)
        if(len(matching_recipes) == 0):
            temp = 0
            #print('NO MATCH FOUND!!!!!')
        else:
            count += 1
        matching_recipes = matching_recipes.difference(set(actual_recipes))
        #print('Found {} recipes with threshold of {}'.format(len(matching_recipes), threshold))
        #print(matching_recipes)

        common_recipes = set(predicted_recipes).intersection(matching_recipes)
        #print('Number of common recipes: {}'.format(len(common_recipes)))
        #print('Accuracy Percentage: {}%'.format(100 * (len(common_recipes)/float(len(predicted_recipes)))))
        #print('List of common recipes: ')
        #print(common_recipes)

        sum += 100 * (len(common_recipes)/float(len(predicted_recipes)))

        #print(data.loc[row['user_id'],:])
        #predicted_recipes, user_df = predicted_recipes = model.recommend(data['user']['user_id'], sparse_user_item, N = 5)

        #if predicted_recipes is None:
        #    continue
    print('Total Number of Customers Tested: {}'.format(len(valid)))
    print('Average Accuracy is: {}'.format(sum/count))
    #print("Total time taken by the process: %s" % str(datetime.now() - default_start_time)[:-3])
    time_in_secs = (datetime.now() - default_start_time).total_seconds()
    print('Total time in seconds: {}'.format(time_in_secs))
    print('Average Time Per Customer: {} seconds'.format(time_in_secs/float(len(valid))))
