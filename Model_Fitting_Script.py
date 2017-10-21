
# coding: utf-8
#!/usr/bin/python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, precision_score, recall_score, f1_score
import numpy as np
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
import time
import sys

__author__="Jonathan Hilgart"




# ### Data Model
# ``` 'loan_amnt',
#  'funded_amnt',
#  'int_rate',
#  'installment',
#  'emp_length',
#  'annual_inc',
#  'zip_code',
#  'dti',
#  'delinq_2yrs',
#  'inq_last_6mths',
#  'mths_since_last_delinq',
#  'mths_since_last_record',
#  'open_acc',
#  'pub_rec',
#  'revol_bal',
#  'revol_util',
#  'total_acc',
#  'collections_12_mths_ex_med',
#  'mths_since_last_major_derog',
#  'acc_now_delinq',
#  'tot_coll_amt',
#  'tot_cur_bal',
#  'open_acc_6m',
#  'open_il_6m',
#  'open_il_12m',
#  'open_il_24m',
#  'total_bal_il',
#  'il_util',
#  'max_bal_bc',
#  'all_util',
#  'total_rev_hi_lim',
#  'inq_last_12m',
#  'acc_open_past_24mths',
#  'avg_cur_bal',
#  'bc_util',
#  'chargeoff_within_12_mths',
#  'delinq_amnt',
#  'mo_sin_old_il_acct',
#  'mo_sin_old_rev_tl_op',
#  'mo_sin_rcnt_rev_tl_op',
#  'mo_sin_rcnt_tl',
#  'mort_acc',
#  'mths_since_recent_bc',
#  'mths_since_recent_bc_dlq',
#  'mths_since_recent_inq',
#  'mths_since_recent_revol_delinq',
#  'num_accts_ever_120_pd',
#  'num_actv_bc_tl',
#  'num_actv_rev_tl',
#  'num_bc_sats',
#  'num_bc_tl',
#  'num_il_tl',
#  'num_op_rev_tl',
#  'num_rev_accts',
#  'num_rev_tl_bal_gt_0',
#  'num_sats',
#  'num_tl_120dpd_2m',
#  'num_tl_30dpd',
#  'num_tl_90g_dpd_24m',
#  'num_tl_op_past_12m',
#  'pct_tl_nvr_dlq',
#  'percent_bc_gt_75',
#  'pub_rec_bankruptcies',
#  'tax_liens',
#  'tot_hi_cred_lim',
#  'total_bal_ex_mort',
#  'total_bc_limit',
#  'total_il_high_credit_limit',
#  'month',
#  'term_ 36 months',
#  'term_ 60 months',
#  'grade_A',
#  'grade_B',
#  'grade_C',
#  'grade_D',
#  'grade_E',
#  'grade_F',
#  'grade_G',
#  'home_ownership_ANY',
#  'home_ownership_MORTGAGE',
#  'home_ownership_NONE',
#  'home_ownership_OTHER',
#  'home_ownership_OWN',
#  'home_ownership_RENT',
#  'purpose_car',
#  'purpose_credit_card',
#  'purpose_debt_consolidation',
#  'purpose_educational',
#  'purpose_home_improvement',
#  'purpose_house',
#  'purpose_major_purchase',
#  'purpose_medical',
#  'purpose_moving',
#  'purpose_other',
#  'purpose_renewable_energy',
#  'purpose_small_business',
#  'purpose_vacation',
#  'purpose_wedding',
#  'addr_state_AK',
#  'addr_state_AL',
#  'addr_state_AR',
#  'addr_state_AZ',
#  'addr_state_CA',
#  'addr_state_CO',
#  'addr_state_CT',
#  'addr_state_DC',
#  'addr_state_DE',
#  'addr_state_FL',
#  'addr_state_GA',
#  'addr_state_HI',
#  'addr_state_IA',
#  'addr_state_ID',
#  'addr_state_IL',
#  'addr_state_IN',
#  'addr_state_KS',
#  'addr_state_KY',
#  'addr_state_LA',
#  'addr_state_MA',
#  'addr_state_MD',
#  'addr_state_ME',
#  'addr_state_MI',
#  'addr_state_MN',
#  'addr_state_MO',
#  'addr_state_MS',
#  'addr_state_MT',
#  'addr_state_NC',
#  'addr_state_ND',
#  'addr_state_NE',
#  'addr_state_NH',
#  'addr_state_NJ',
#  'addr_state_NM',
#  'addr_state_NV',
#  'addr_state_NY',
#  'addr_state_OH',
#  'addr_state_OK',
#  'addr_state_OR',
#  'addr_state_PA',
#  'addr_state_RI',
#  'addr_state_SC',
#  'addr_state_SD',
#  'addr_state_TN',
#  'addr_state_TX',
#  'addr_state_UT',
#  'addr_state_VA',
#  'addr_state_VT',
#  'addr_state_WA',
#  'addr_state_WI',
#  'addr_state_WV',
#  'addr_state_WY']'```

# In[8]:


# In[9]:

def train_test_split_val(input_df):
    """Create train test validation splits
    Test is the test set"""

    y_col = input_df.repaid
    x_cols = input_df.iloc[:,input_df.columns!='repaid']
    #scale the x_variables
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_cols)
    X_train, X_test, y_train, y_test = train_test_split(x_scaled, y_col,test_size=.15)
    X_t, X_val, y_t, y_val = train_test_split(X_train, y_train,test_size=.15)
    return X_t,X_train,X_val, y_t, y_train, y_val



def fit_rf(input_x,input_y, n_trees):
    """Fit a random forest with the number of trees given"""
    rf = RandomForestClassifier(n_estimators=n_trees ,n_jobs=-1, verbose = 1)
    # Time training
    s_rf = time.time()
    sys.stderr.write("Fitting Random Forest \n")
    rf.fit(X_t,y_t)
    e_rf = time.time()
    sys.stderr.write("RF training took {} minutes \n".format((e_rf-s_rf)/60))
    return rf


def rf_metrics(val_x, val_y, rf, n_trees):
    """Determine the metrics for the RF.
    returns: accuracy, recall, precision, F1"""

    predictions_rf = rf.predict(val_x)
    positive_probability_predictions = rf.predict_proba(X_val)[:,1]
    accuracy_rf = accuracy_score(val_y, predictions_rf)
    recall_rf = recall_score(val_y, predictions_rf)
    precision_rf = precision_score(val_y, predictions_rf)
    f1_rf = f1_score(val_y, predictions_rf)
    sys.stderr.write("Accuracy of Random Forrest with {} trees was {:.2%} \n".format(n_trees, accuracy_rf ))
    sys.stderr.write("recall of RF is {:.2%} \n".format(recall_rf))
    sys.stderr.write("Precision of RF is {:.2%} \n".format(precision_rf))
    sys.stderr.write("F1 score of RF was {:.2%} \n".format(f1_rf))
    return accuracy_rf, recall_rf, precision_rf, f1_rf


def fit_gb(input_x,input_y, n_trees):
    """Fit a gradient boosting classifier with the given number of trees"""
    gb = GradientBoostingClassifier(n_estimators=n_trees, verbose=1)
    s_gb = time.time()
    sys.stderr.write("Fitting GB \n")
    gb.fit(input_x, input_y)
    e_gb  = time.time()
    sys.stderr.write("GB training took {} \n".format((e_gb-s_gb)/60))
    return gb

def gb_metrics(gb, x_val, y_val , n_trees):
    """sys.stderr.write out the metrics for GB classifier
    Returns: accuracy, recall, precision, f1"""
    predictions_gb = gb.predict(x_val)
    positive_probability_predictions_gb = gb.predict_proba(x_val)[:,1]
    accuracy_gb = accuracy_score(y_val, predictions_gb)
    recall_gb = recall_score(y_val, predictions_gb)
    precision_gb = precision_score(y_val, predictions_gb)
    f1_gb = f1_score(y_val, predictions_gb)
    sys.stderr.write("Accuracy of GB with {} trees was {} \n".format(n_trees,accuracy_gb))
    sys.stderr.write("recall of GB is {:.2%} \n".format(recall_gb))
    sys.stderr.write("Precision of GB is {:.2%} \n".format(precision_gb))
    sys.stderr.write("F1 score of GB was {:.2%} \n".format(f1_gb))
    return accuracy_gb, recall_gb, precision_gb, f1_gb






if __name__ =="__main__":
    # bring in CLI arguments
    if len(sys.argv)!=4:
        sys.stderr.write("You need to include three arguments: num_trees_rf, num_trees_gb, cv_times")
        sys.exit() # exit script
    num_trees_rf = int(sys.argv[1])
    num_trees_gb = int(sys.argv[2])
    cv_times = int(sys.argv[3])



    #Load data
    sys.stderr.write('Loading data \n')
    lending_club_df = pd.read_csv("./data/final_lending_club.csv")
    sys.stderr.write('Finished loading data \n')
    # Ensure DataFrame model is the same as the data model
    lending_club_df_training = lending_club_df[['loan_amnt','repaid',
 'funded_amnt','int_rate','installment','emp_length','annual_inc','zip_code','dti','delinq_2yrs','inq_last_6mths',
 'mths_since_last_delinq','mths_since_last_record','open_acc','pub_rec',
 'revol_bal','revol_util','total_acc','collections_12_mths_ex_med',
 'mths_since_last_major_derog', 'acc_now_delinq', 'tot_coll_amt', 'tot_cur_bal', 'open_acc_6m','open_il_6m', 'open_il_12m','open_il_24m',
 'total_bal_il','il_util','max_bal_bc','all_util', 'total_rev_hi_lim','inq_last_12m','acc_open_past_24mths',
 'avg_cur_bal', 'bc_util', 'chargeoff_within_12_mths', 'delinq_amnt','mo_sin_old_il_acct', 'mo_sin_old_rev_tl_op',
 'mo_sin_rcnt_rev_tl_op', 'mo_sin_rcnt_tl', 'mort_acc', 'mths_since_recent_bc', 'mths_since_recent_bc_dlq','mths_since_recent_inq',
 'mths_since_recent_revol_delinq', 'num_accts_ever_120_pd', 'num_actv_bc_tl', 'num_actv_rev_tl', 'num_bc_sats','num_bc_tl' , 'num_il_tl',
 'num_op_rev_tl', 'num_rev_accts', 'num_rev_tl_bal_gt_0', 'num_sats', 'num_tl_120dpd_2m', 'num_tl_30dpd', 'num_tl_90g_dpd_24m','num_tl_op_past_12m','pct_tl_nvr_dlq',
 'percent_bc_gt_75','pub_rec_bankruptcies', 'tax_liens',
 'tot_hi_cred_lim','total_bal_ex_mort', 'total_bc_limit','total_il_high_credit_limit','month', 'term_ 36 months','term_ 60 months',
 'grade_A', 'grade_B', 'grade_C', 'grade_D', 'grade_E', 'grade_F','grade_G','home_ownership_ANY','home_ownership_MORTGAGE','home_ownership_NONE',
 'home_ownership_OTHER','home_ownership_OWN', 'home_ownership_RENT',
 'purpose_car','purpose_credit_card','purpose_debt_consolidation','purpose_educational',
 'purpose_home_improvement','purpose_house','purpose_major_purchase','purpose_medical','purpose_moving','purpose_other','purpose_renewable_energy',
 'purpose_small_business','purpose_vacation','purpose_wedding','addr_state_AK','addr_state_AL',
 'addr_state_AR' ,'addr_state_AZ' ,'addr_state_CA','addr_state_CO',
 'addr_state_CT','addr_state_DC',
 'addr_state_DE', 'addr_state_FL','addr_state_GA','addr_state_HI', 'addr_state_IA',
 'addr_state_ID','addr_state_IL', 'addr_state_IN','addr_state_KS',
 'addr_state_KY', 'addr_state_LA','addr_state_MA','addr_state_MD','addr_state_ME','addr_state_MI', 'addr_state_MN','addr_state_MO','addr_state_MS','addr_state_MT',
 'addr_state_NC','addr_state_ND','addr_state_NE','addr_state_NH','addr_state_NJ','addr_state_NM','addr_state_NV','addr_state_NY','addr_state_OH','addr_state_OK','addr_state_OR','addr_state_PA','addr_state_RI','addr_state_SC',
 'addr_state_SD','addr_state_TN','addr_state_TX','addr_state_UT','addr_state_VA','addr_state_VT','addr_state_WA','addr_state_WI',
 'addr_state_WV',
 'addr_state_WY']]
    # train test validate CV
    ## Keep track of CV metrics
    ### RF
    rf_accuracy_cv = []
    rf_recall_cv = []
    rf_precision_cv = []
    rf_f1_cv = []
    ### GB
    gb_accuracy_cv = []
    gb_recall_cv = []
    gb_precision_cv = []
    gb_f1_cv = []

    for  _ in range(cv_times):
        X_t,X_train,X_val, y_t, y_train, y_val = train_test_split_val(lending_club_df_training )
        # fit RF
        rf = fit_rf(X_t,y_t, num_trees_rf)
        # Determine metrics
        accuracy_rf, recall_rf, precision_rf, f1_rf = rf_metrics(X_val, y_val, rf, num_trees_rf)
        rf_accuracy_cv.append(accuracy_rf)
        rf_recall_cv.append(recall_rf)
        rf_precision_cv.append(precision_rf)
        rf_f1_cv.append(f1_rf)
        # Fit GB
        gb = fit_gb(X_t,y_t,num_trees_gb)
        # determine gb metrics
        accuracy_gb, recall_gb, precision_gb, f1_gb = gb_metrics(gb, X_val,y_val, num_trees_gb)
        gb_accuracy_cv.append(accuracy_gb)
        gb_recall_cv.append(recall_gb)
        gb_precision_cv.append(precision_gb)
        gb_f1_cv.append(f1_gb)
    sys.stderr.write("############################### \n")
    sys.stderr.write("FINISHED TRAINING \n")
    sys.stderr.write("############################### \n")
    sys.stderr.write(' ~ Random Forest ~ \n')
    sys.stderr.write('Final RF accuracy {} \n'.format(np.mean(rf_accuracy_cv )))
    sys.stderr.write('Final RF recall {} \n'.format(np.mean(rf_recall_cv)))
    sys.stderr.write('Final RF precision {} \n'.format(np.mean(rf_precision_cv)))
    sys.stderr.write('Final RF f1 \n' .format(np.mean(rf_f1_cv)))
    sys.stderr.write('~ Gradient Boosting ~ \n')
    sys.stderr.write('Final GB accuracy {} \n'.format(np.mean(gb_accuracy_cv )))
    sys.stderr.write('Final GB recall {} \n'.format(np.mean(gb_recall_cv)))
    sys.stderr.write('Final GB precision {} \n'.format(np.mean(gb_precision_cv)))
    sys.stderr.write('Final GB f1 \n'.format(np.mean(gb_f1_cv)))
    # SAVE THE MODELS
    filename = './models/random_forest_{}-trees.joblib.pkl'.format(num_trees_rf)
    _ = joblib.dump(rf, filename, compress=9)

    filename = './models/gradient_boosting_{}-trees.joblib.pkl'.format(num_trees_gb)
    _ = joblib.dump(gb, filename, compress=9)
