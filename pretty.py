import pandas as pd
import numpy as np
from time import strftime, localtime

def get_timestamped_file_name(filename, path='./', postfix="csv"):
    '''
    description: get a file name with time stamp
    example:
        filename : wanfang_authors
        prosfix  : csv
    return : wanfang_authors_2016_08_22_05_52_51.csv
    '''
    return "%s/%s_%s.%s" % (path, filename, strftime("%Y_%m_%d_%H_%M_%S", localtime()), postfix)

def create_submission(confidence_scores, save_path):
    submission = pd.DataFrame({"score": confidence_scores})
    submission.to_csv(save_path, index_label="id")

def pretty(submits, up, low):

    df = pd.read_csv(submits)

    df.loc[df['score'] > up, 'score'] = 1
    df.loc[df['score'] < low, 'score'] = 0

    submit_file = get_timestamped_file_name('./{}_{}_{}_pretty'.format(submits, up, low), './', 'csv')
    df.to_csv(submit_file, index=False)



#  0.96509   ["./predict/submission6.csv", 0.99, 0.01]
#  0.96103   ["./predict/submission6.csv", 0.99, 0.01]

pretty("./predict/submission6.csv", 0.98, 0)
