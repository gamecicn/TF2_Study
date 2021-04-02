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

def blend(submits):
    df = pd.concat([pd.read_csv(x)['score'] for x in submits], axis=1)
    submit_file = get_timestamped_file_name('submission', './', 'csv')
    create_submission(df.mean(axis=1), submit_file)



MIX_1 = [ # 96283
    "./predict/submission2.csv",
    './predict/submission_2021_03_31_09_44_37.csv'
]

MIX_2 = [ # 95617
    "./predict/submission3.csv",
    './predict/submission5.csv'
]

MIX_3 = [ # 0.96412
    "./predict/submission3.csv",
    './predict/submission2.csv'
]

MIX_4 = [ # 0.96379
    "./predict/submission4.csv",
    './predict/submission2.csv'
]

MIX_5 = [ # 0.96379
    "./predict/submission4.csv",
    './predict/submission2.csv'
]

MIX_6 = [ # 0.
    "./predict/submission4.csv",
    './predict/submission_2021_03_31_13_20_23.csv'
]

MIX_7 = [ # 0.96508  ===>submission6.csv
    "./predict/submission4.csv",
    './predict/submission_xgb_3.csv'
]

MIX_8 = [ # 0.96408
    "./predict/submission4.csv",
    './predict/submission_xgb_5.csv'
]

MIX_9 = [ # 0.96446
    "./predict/submission6.csv",
    './predict/submission_xgb_5.csv'
]


blend(MIX_9)












