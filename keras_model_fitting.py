#! usr/bin/env/python
import pandas as pd
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,  roc_curve,  precision_score,  recall_score,  f1_score
import numpy as np
from keras.layers import Dense,  Dropout
from keras.constraints import maxnorm
from keras.optimizers import Adam
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import load_model
from keras.callbacks import LearningRateScheduler
import sys
import math
__author__="Jonathan Hilgart"


def split_scale_data(input_df):
    """Split and whiten the input data"""
    y_col = input_df.repaid
    x_cols = lending_club_df_training.loc[:,lending_club_df.columns[lending_club_df.columns!='repaid'] ]
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_cols)
    X_train,  X_test,  y_train,  y_test = train_test_split(x_scaled,
                                                    y_col, test_size=.05)
    return X_train,  X_test,  y_train,  y_test


def build_model(learning_rate=.15, decay_rate=0.0):
    """Build a Keras binary classifier model"""
    model = Sequential()
    model.add(Dense(units=100,  input_shape=(149, ),  activation='relu',
                    kernel_constraint=maxnorm(4)))
    model.add(Dropout(.2))
    model.add(Dense(units=300,   activation='relu', kernel_constraint=maxnorm(4)))
    model.add(Dropout(.25))
    model.add(Dense(units=500,   activation='relu', kernel_constraint=maxnorm(4)))
    model.add(Dropout(.3))
    model.add(Dense(units=300,   activation='relu', kernel_constraint=maxnorm(4)))
    model.add(Dropout(.2))
    model.add(Dense(units=100,   activation='relu', kernel_constraint=maxnorm(4)))
    model.add(Dense(units=1,   activation='sigmoid'))
    adam = Adam(lr=learning_rate, decay=decay_rate)
    model.compile(optimizer=adam,  loss='binary_crossentropy',  metrics=['accuracy'])
    return model



if __name__ == "__main__":
    # Rad in system arguments
    if len(sys.argv) != 3:
        sys.stderr.write("You need to pass in two additional arguments. \
                         1: The number of epochs,  2: The batch size\n")
        sys.exit()  # exit script
    epo = int(sys.argv[1])
    batch_size_in = int(sys.argv[2])
    # adaptive decay
    learning_r = 0.01
    decay_r = learning_r / epo
    # read in data
    sys.stderr.write("Reading in data\n")
    lending_club_df = pd.read_csv("./data/final_lending_club.csv")
    sys.stderr.write("Finished readingin data\n")
    # make sure data is in the correct order
    lending_club_df_training = lending_club_df[['loan_amnt',
    'funded_amnt', 'int_rate', 'installment', 'emp_length', 'annual_inc', 'zip_code', 'dti',
    'repaid', 'delinq_2yrs', 'inq_last_6mths', 'mths_since_last_delinq',
    'mths_since_last_record', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 'collections_12_mths_ex_med', 'mths_since_last_major_derog', 'acc_now_delinq',
    'tot_coll_amt',
    'tot_cur_bal', 'open_acc_6m', 'open_il_6m', 'open_il_12m', 'open_il_24m', 'total_bal_il', 'il_util',
    'max_bal_bc' , 'all_util', 'total_rev_hi_lim', 'inq_last_12m', 'acc_open_past_24mths', 'avg_cur_bal', 'bc_util',
    'chargeoff_within_12_mths',
    'delinq_amnt', 'mo_sin_old_il_acct', 'mo_sin_old_rev_tl_op', 'mo_sin_rcnt_rev_tl_op', 'mo_sin_rcnt_tl', 'mort_acc', 'mths_since_recent_bc', 'mths_since_recent_bc_dlq', 'mths_since_recent_inq', 'mths_since_recent_revol_delinq', 'num_accts_ever_120_pd',
    'num_actv_bc_tl',
    'num_actv_rev_tl', 'num_bc_sats', 'num_bc_tl', 'num_il_tl', 'num_op_rev_tl', 'num_rev_accts', 'num_rev_tl_bal_gt_0', 'num_sats', 'num_tl_120dpd_2m',
    'num_tl_30dpd',
    'num_tl_90g_dpd_24m', 'num_tl_op_past_12m', 'pct_tl_nvr_dlq', 'percent_bc_gt_75', 'pub_rec_bankruptcies',
    'tax_liens', 'tot_hi_cred_lim', 'total_bal_ex_mort', 'total_bc_limit',
    'total_il_high_credit_limit',
    'month',
    'term_ 36 months',
    'term_ 60 months', 'grade_A', 'grade_B', 'grade_C', 'grade_D',  'grade_E',
    'grade_F', 'grade_G', 'home_ownership_ANY', 'home_ownership_MORTGAGE',
    'home_ownership_NONE', 'home_ownership_OTHER', 'home_ownership_OWN', 'home_ownership_RENT', 'purpose_car', 'purpose_credit_card', 'purpose_debt_consolidation', 'purpose_educational', 'purpose_home_improvement',
    'purpose_house' , 'purpose_major_purchase',  'purpose_medical', 'purpose_moving', 'purpose_other', 'purpose_renewable_energy', 'purpose_small_business',
    'purpose_vacation', 'purpose_wedding', 'addr_state_AK', 'addr_state_AL',
    'addr_state_AR', 'addr_state_AZ', 'addr_state_CA',
    'addr_state_CO',
    'addr_state_CT',
    'addr_state_DC',
    'addr_state_DE', 'addr_state_FL', 'addr_state_GA', 'addr_state_HI',
    'addr_state_IA', 'addr_state_ID', 'addr_state_IL',
    'addr_state_IN', 'addr_state_KS', 'addr_state_KY', 'addr_state_LA',
    'addr_state_MA', 'addr_state_MD', 'addr_state_ME',
    'addr_state_MI', 'addr_state_MN',
    'addr_state_MO', 'addr_state_MS', 'addr_state_MT', 'addr_state_NC',
    'addr_state_ND', 'addr_state_NE', 'addr_state_NH', 'addr_state_NJ', 'addr_state_NM',
    'addr_state_NV', 'addr_state_NY', 'addr_state_OH',
    'addr_state_OK',
    'addr_state_OR', 'addr_state_PA', 'addr_state_RI', 'addr_state_SC', 'addr_state_SD', 'addr_state_TN',
    'addr_state_TX', 'addr_state_UT', 'addr_state_VA', 'addr_state_VT', 'addr_state_WA', 'addr_state_WI', 'addr_state_WV',
    'addr_state_WY']]
    # Split and scale the data
    sys.stderr.write("Finished reading data. NOw scaling and splitting data\n")
    X_train,  X_test,  y_train,  y_test = split_scale_data(lending_club_df_training)
    sys.stderr.write("Finished splitting/scaling data\n")
    # Build the model
    m = build_model(learning_rate=learning_r, decay_rate=decay_r)
    # Fit the model with decaying learning rate
    history = m.fit(X_train, y_train,  epochs=epo, batch_size=batch_size_in,
        validation_split=.05,  verbose=1)
    # Print out the results
    for k, v in history.history.items():
        # print out the results from training
        sys.stderr.write("\n")
        sys.stderr.write("{} - {}".format(k,v))
        sys.stderr.write("\n")
    # get the final accuracy
    train_acc = history.history['acc'][-1]
    val_acc = history.history['val_acc'][-1]
    if train_acc > .80:  # only save if high train accuracy
        # Save the model
        m.save('models/mlp_model_{}-epochs_{}-batchsize_{}-train_acc_{}-val_acc.h5'.format(epo, batch_size_in, train_acc, val_acc)) # save weigts to h5py
        # serialize model to JSON
        model_json=m.to_json()
        with open('models/mlp_model_{}-epochs_{}-batchsize_{}-train_acc_{}-val_acc'.format(epo, batch_size_in, train_acc, val_acc),  "w") as json_file:
            json_file.write(model_json)
