import pandas as pd
import numpy as np
import argparse

def split_data(split_ratio, valid_ratio, input_file):
    '''
    #############################################
    data = pd.read_csv(input_file)
    data.columns = ['user', 'recipe', 'counts']

    data['user'] = data['user'].astype("category")
    data['recipe'] = data['recipe'].astype("category")
    data['user_id'] = data['user'].cat.codes
    data['recipe_id'] = data['recipe'].cat.codes

    user_lookup = data[['user_id', 'user']].drop_duplicates()
    item_lookup = data[['recipe_id', 'recipe']].drop_duplicates()

    data = data.infer_objects()

    #print(data.columns)
    #print(data.dtypes)

    total_sum = data.counts.sum()

    print('Total Transactions: {}'.format(len(data)))
    print('Total purchases: {}'.format(total_sum))
    print('Total unique Users: {}'.format(len(user_lookup)))
    print('Total unique Recipes: {}'.format(len(item_lookup)))

    #############################################
    '''
    df = pd.read_csv(input_file)
    df['split'] = np.random.randn(df.shape[0], 1)

    msk = np.random.rand(len(df)) <= split_ratio

    train = df[msk]
    test = df[~msk]

    train.iloc[:,0:3].to_csv('train.csv', index=False)
    test.iloc[:,0:3].to_csv('test.csv', index=False)

    validation_set = train.iloc[:,0:3]
    validation_set['split'] = np.random.randn(validation_set.shape[0], 1)

    msk = np.random.rand(len(validation_set)) <= valid_ratio
    valid = validation_set[msk]

    s = pd.DataFrame(valid.iloc[:,0].unique())
    s.columns = ['user_id']
    s.to_csv('valid.csv', index=False)

    print('Number of Training Rows: {}'.format(len(train)))
    print('Number of Testing Rows: {}'.format(len(test)))
    print('Validating for {} users'.format(len(s)))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Splitting Customer Purchase data')
    required_args = parser.add_argument_group('required arguments')

    required_args.add_argument(
        "-i", "--input-file", dest="input_file", required=True, \
        help="Input file with customer purchase data, Currently only .csv files are supported"
    )

    required_args.add_argument(
        "-s", "--split-ratio", dest="split_ratio", type=float, required=True, \
        help="This ratio is used to split input data into training and testing dataset"
    )

    required_args.add_argument(
        "-v", "--valid-ratio", dest="valid_ratio", type=float, required=True, \
        help="This ratio is used to split input data into training and testing dataset"
    )
    arguments = parser.parse_args()
    split_data(arguments.split_ratio, arguments.valid_ratio, arguments.input_file)
