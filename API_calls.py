#!/usr/bin/python
# coding: utf-8
import requests
import os
import yaml
import pandas as pd
import numpy as np
import re
import pickle
from sklearn.externals import joblib
import json
import datetime
import sys

BEST_MODEL_FILE = './models/gradient_boosting_400-trees.joblib.pkl'


def check_cash(credentials):

    # # Check available cash
    # ```
    # URL: https://api.lendingclub.com/api/investor/<version>/accounts/<investor id>/availablecash
    # ```


    cash_r = requests.get('https://api.lendingclub.com/api/investor/v1/accounts/{}/availablecash'.format(account_id),
                     headers={'Authorization': credentials['authentication'],'Content-Type': 'application/json',
                              'Accept': 'application/json'}, params = {'showAll':'true'})
    available_cash = cash_r.json()['availableCash'] # see if available cash > $25
    return available_cash


def get_loans(credentials):
    """Get the available loans on the lending club platform. Convert to a pandas DF
    and return the DF"""
    r = requests.get('https://api.lendingclub.com/api/investor/v1/loans/listing',
                     headers={'Authorization': credentials['authentication'], 'Content-Type': 'application/json',
                              'Accept': 'application/json'}, params={'showAll':'true'})
    json_response = r.json()
    # convert to DF
    loans_df = pd.DataFrame(json_response['loans'])
    return loans_df


def create_feature_columns(loans_df):
    """Wrangle the loans from the API to match what the gradient boosting model
    was trained on. Return the new final DF"""
    # running into problems with openIl6m not being returned - convert to all zeros if this is true
    if 'openIl6m' in loans_df.columns:
        pass
    else:
        loans_df['openIl6m'] = 0
    # #### Convert EmpLength to years
    loans_df.empLength = loans_df.empLength.apply(lambda x: x / 60 if x > 60 else 0)
    # #### Create month to int 1-12
    loans_df['month'] = loans_df.acceptD.apply(lambda x: pd.to_datetime(x).month)
    # #### Crea cols term_36 and term_60
    loans_df['term_36'] = [1 if i == 36 else 0 for i in loans_df.term]
    loans_df['term_60'] = [1 if i == 60 else 0 for i in loans_df.term]
    # #### Create grade columns
    loans_df['grade_a'] = [1 if i == 'A' else 0 for i in loans_df.grade]
    loans_df['grade_b'] = [1 if i == 'B' else 0 for i in loans_df.grade]
    loans_df['grade_c'] = [1 if i == 'C' else 0 for i in loans_df.grade]
    loans_df['grade_d'] = [1 if i == 'D' else 0 for i in loans_df.grade]
    loans_df['grade_e'] = [1 if i == 'E' else 0 for i in loans_df.grade]
    loans_df['grade_f'] = [1 if i == 'F' else 0 for i in loans_df.grade]
    loans_df['grade_g'] = [1 if i == 'G' else 0 for i in loans_df.grade]
    # ### Create home ownership columns
    loans_df['home_ownership_any'] = [1 if i == 'ANY' else 0 for i in loans_df.homeOwnership]
    loans_df['home_ownership_mortgage'] = [1 if i == 'MORTGAGE' else 0 for i in loans_df.homeOwnership]
    loans_df['home_ownership_none'] = [1 if i == 'NONE' else 0 for i in loans_df.homeOwnership]
    loans_df['home_ownership_other'] = [1 if i == 'OTHER' else 0 for i in loans_df.homeOwnership]
    loans_df['home_ownership_own'] = [1 if i == 'OWN' else 0 for i in loans_df.homeOwnership]
    loans_df['home_ownership_rent'] = [1 if i == 'RENT' else 0 for i in loans_df.homeOwnership]
    # ## Create purpose columns
    loans_df['purpose_credit_card'] = [1 if i =='credit_card' else 0 for i in loans_df.purpose]
    loans_df['purpose_debt_consolidation'] = [1 if i =='debt_consolidation' else 0 for i in loans_df.purpose]
    loans_df['purpose_educational'] = [1 if i =='educational' else 0 for i in loans_df.purpose]
    loans_df['purpose_home_improvement'] = [1 if i =='home_improvement' else 0 for i in loans_df.purpose]
    loans_df['purpose_house'] = [1 if i =='house' else 0 for i in loans_df.purpose]
    loans_df['purpose_major_purchase'] = [1 if i =='major_purchase' else 0 for i in loans_df.purpose]
    loans_df['purpose_medical'] = [1 if i =='medical' else 0 for i in loans_df.purpose]
    loans_df['purpose_moving'] = [1 if i =='moving' else 0 for i in loans_df.purpose]
    loans_df['purpose_other'] = [1 if i =='other' else 0 for i in loans_df.purpose]
    loans_df['purpose_renewable_energy'] = [1 if i =='renewable_energy' else 0 for i in loans_df.purpose]
    loans_df['purpose_small_business'] = [1 if i =='small_business' else 0 for i in loans_df.purpose]
    loans_df['purpose_vacation'] = [1 if i =='vacation' else 0 for i in loans_df.purpose]
    loans_df['purpose_wedding'] = [1 if i =='wedding' else 0 for i in loans_df.purpose]
    loans_df['purpose_car'] = [1 if i =='car' else 0 for i in loans_df.purpose]
    # ### Create the states columns
    loans_df['addr_state_AK'] = [1 if i == 'AK' else 0 for i in loans_df.addrState]
    loans_df['addr_state_AL'] = [1 if i == 'AL' else 0 for i in loans_df.addrState]
    loans_df['addr_state_AR'] = [1 if i == 'AR' else 0 for i in loans_df.addrState]
    loans_df['addr_state_AZ'] = [1 if i == 'AZ' else 0 for i in loans_df.addrState]
    loans_df['addr_state_CA'] = [1 if i == 'CA' else 0 for i in loans_df.addrState]
    loans_df['addr_state_CO'] = [1 if i == 'CO' else 0 for i in loans_df.addrState]
    loans_df['addr_state_CT'] = [1 if i == 'CT' else 0 for i in loans_df.addrState]
    loans_df['addr_state_DC'] = [1 if i == 'DC' else 0 for i in loans_df.addrState]
    loans_df['addr_state_DE'] = [1 if i == 'DE' else 0 for i in loans_df.addrState]
    loans_df['addr_state_FL'] = [1 if i == 'FL' else 0 for i in loans_df.addrState]
    loans_df['addr_state_GA'] = [1 if i == 'GA' else 0 for i in loans_df.addrState]
    loans_df['addr_state_HI'] = [1 if i == 'HI' else 0 for i in loans_df.addrState]
    loans_df['addr_state_IA'] = [1 if i == 'IA' else 0 for i in loans_df.addrState]
    loans_df['addr_state_ID'] = [1 if i =='ID' else 0 for i in loans_df.addrState]
    loans_df['addr_state_IL'] = [1 if i =='IL' else 0 for i in loans_df.addrState]
    loans_df['addr_state_IN'] = [1 if i =='IN' else 0 for i in loans_df.addrState]
    loans_df['addr_state_KS'] = [1 if i =='KS' else 0 for i in loans_df.addrState]
    loans_df['addr_state_KY'] = [1 if i =='KY' else 0 for i in loans_df.addrState]
    loans_df['addr_state_LA'] = [1 if i =='LA' else 0 for i in loans_df.addrState]
    loans_df['addr_state_MA'] = [1 if i =='MA' else 0 for i in loans_df.addrState]
    loans_df['addr_state_MD'] = [1 if i =='MD' else 0 for i in loans_df.addrState]
    loans_df['addr_state_ME'] = [1 if i =='ME' else 0 for i in loans_df.addrState]
    loans_df['addr_state_MI'] = [1 if i =='MI' else 0 for i in loans_df.addrState]
    loans_df['addr_state_MN'] = [1 if i =='MN' else 0 for i in loans_df.addrState]
    loans_df['addr_state_MO'] = [1 if i =='MO' else 0 for i in loans_df.addrState]
    loans_df['addr_state_MS'] = [1 if i =='MS' else 0 for i in loans_df.addrState]
    loans_df['addr_state_MT'] = [1 if i =='MT' else 0 for i in loans_df.addrState]
    loans_df['addr_state_NC'] = [1 if i =='NC' else 0 for i in loans_df.addrState]
    loans_df['addr_state_ND'] = [1 if i =='ND' else 0 for i in loans_df.addrState]
    loans_df['addr_state_NE'] = [1 if i =='NE' else 0 for i in loans_df.addrState]
    loans_df['addr_state_NH'] = [1 if i =='NH' else 0 for i in loans_df.addrState]
    loans_df['addr_state_NJ'] = [1 if i =='NJ' else 0 for i in loans_df.addrState]
    loans_df['addr_state_NM'] = [1 if i =='NM' else 0 for i in loans_df.addrState]
    loans_df['addr_state_NV'] = [1 if i =='NV' else 0 for i in loans_df.addrState]
    loans_df['addr_state_NY'] = [1 if i =='NY' else 0 for i in loans_df.addrState]
    loans_df['addr_state_OH'] = [1 if i =='OH' else 0 for i in loans_df.addrState]
    loans_df['addr_state_OK'] = [1 if i =='OK' else 0 for i in loans_df.addrState]
    loans_df['addr_state_OR'] = [1 if i =='OR' else 0 for i in loans_df.addrState]
    loans_df['addr_state_PA'] = [1 if i =='PA' else 0 for i in loans_df.addrState]
    loans_df['addr_state_RI'] = [1 if i =='RI' else 0 for i in loans_df.addrState]
    loans_df['addr_state_SC'] = [1 if i =='SC' else 0 for i in loans_df.addrState]
    loans_df['addr_state_SD'] = [1 if i =='SD' else 0 for i in loans_df.addrState]
    loans_df['addr_state_TN'] = [1 if i =='TN' else 0 for i in loans_df.addrState]
    loans_df['addr_state_TX'] = [1 if i =='TX' else 0 for i in loans_df.addrState]
    loans_df['addr_state_UT'] = [1 if i =='UT' else 0 for i in loans_df.addrState]
    loans_df['addr_state_VA'] = [1 if i =='VA' else 0 for i in loans_df.addrState]
    loans_df['addr_state_VT'] = [1 if i =='VT' else 0 for i in loans_df.addrState]
    loans_df['addr_state_WA'] = [1 if i =='WA' else 0 for i in loans_df.addrState]
    loans_df['addr_state_WI'] = [1 if i =='WI' else 0 for i in loans_df.addrState]
    loans_df['addr_state_WV'] = [1 if i =='WV' else 0 for i in loans_df.addrState]
    loans_df['addr_state_WY'] = [1 if i =='WY' else 0 for i in loans_df.addrState]
    # #### Create the zip code column
    loans_df.addrZip = loans_df.addrZip.apply(  lambda x: float(x[:3]))
    # Columns to match the data model

    api_cols =['loanAmount',
               'fundedAmount',
               'intRate',
               'installment',
               'empLength',
               'annualInc',
               'addrZip',
              'dti',
               'delinq2Yrs',
               'inqLast6Mths',
           'mthsSinceLastDelinq',
               'mthsSinceLastRecord',
               'openAcc',
               'pubRec',
               'revolBal',
               'revolUtil',
               'totalAcc',
          'collections12MthsExMed',
               'mthsSinceLastMajorDerog',
               'accNowDelinq',
               'totCollAmt',
               'totCurBal',
               'openAcc6m',
          'openIl6m',
            'openIl12m',
               'openIl24m',
               'totalBalIl',
               'iLUtil',
               'maxBalBc',
               'allUtil',
            'totalRevHiLim',
               'inqLast12m',
               'accOpenPast24Mths',
          'avgCurBal',
               'bcUtil',
               'chargeoffWithin12Mths',
               'delinqAmnt',
               'moSinOldIlAcct',
               'moSinOldRevTlOp',
               'moSinRcntRevTlOp',
         'moSinRcntTl',
               'mortAcc',
               'mthsSinceRecentBc',
               'mthsSinceRecentBcDlq',
               'mthsSinceRecentInq',
               'mthsSinceRecentRevolDelinq',
          'numAcctsEver120Ppd',
               'numActvBcTl',
               'numActvRevTl',
               'numBcSats',
               'numBcTl',
               'numIlTl',
               'numOpRevTl',
               'numRevAccts',
          'numRevTlBalGt0',
               'numSats',
               'numTl120dpd2m',
               'numTl30dpd',
               'numTl90gDpd24m',
               'numTlOpPast12m',
               'pctTlNvrDlq',
          'percentBcGt75',
               'pubRecBankruptcies',
               'taxLiens',
               'totHiCredLim',
               'totalBalExMort',
               'totalBcLimit',
               'totalIlHighCreditLimit',
          'month',
               'term_36',
               'term_60',
               'grade_a',
               'grade_b',
               'grade_c',
               'grade_d',
               'grade_e',
               'grade_f',
               'grade_g',
          'home_ownership_any','home_ownership_mortgage','home_ownership_none','home_ownership_other','home_ownership_own',
          'home_ownership_rent',
               'purpose_car','purpose_credit_card','purpose_debt_consolidation','purpose_educational','purpose_home_improvement',
          'purpose_house','purpose_major_purchase','purpose_medical','purpose_moving','purpose_other','purpose_renewable_energy',
          'purpose_small_business','purpose_vacation','purpose_wedding',
               'addr_state_AK','addr_state_AL','addr_state_AR',
          'addr_state_AZ','addr_state_CA','addr_state_CO', 'addr_state_CT','addr_state_DC','addr_state_DE','addr_state_FL',
          'addr_state_GA','addr_state_HI','addr_state_IA', 'addr_state_ID','addr_state_IL','addr_state_IN','addr_state_KS',
          'addr_state_KY','addr_state_LA','addr_state_MA', 'addr_state_MD','addr_state_ME','addr_state_MI','addr_state_MN',
          'addr_state_MO','addr_state_MS','addr_state_MT', 'addr_state_NC','addr_state_ND','addr_state_NE','addr_state_NH',
          'addr_state_NJ','addr_state_NM','addr_state_NV', 'addr_state_NY','addr_state_OH','addr_state_OK','addr_state_OR',
          'addr_state_PA','addr_state_RI','addr_state_SC', 'addr_state_SD','addr_state_TN','addr_state_TX','addr_state_UT',
          'addr_state_VA','addr_state_VT','addr_state_WA', 'addr_state_WI','addr_state_WV','addr_state_WY']
    # Creat the final data model
    final_api_df = loans_df[api_cols]
    # ### Impute zeros for NaNs
    final_api_df.fillna(0, inplace=True)
    return final_api_df


def load_model():
    """"Load the trained GB model"""
    loaded_gb_model = joblib.load(open(BEST_MODEL_FILE, 'rb'))
    return loaded_gb_model


def predict_loan_success(loaded_gb_model, final_api_df):
    """Predict if someone will pay back each loan with the trained
    GB model"""
    # classes go [0,1]
    probability_predictions = loaded_gb_model.predict_proba(final_api_df)[:, 1]
    # threshold of repaying to 70%
    loans_to_purchase = loans_df.id[probability_predictions > .65]
    return loans_to_purchase


def purchase_loans(available_cash, account_id, loans_to_purchase,
                   portfolio_id, credentials):
    """Purchases loans if there is more than $25 of available cash and the gradient boosting
    model thinks that the loan will be repaid"""
    count = 0
    if count == len(loans_to_purchase):
        sys.stderr.write(" No loans to purchase at {} \n".format(
            datetime.datetime.today()))
    while count != len(loans_to_purchase):
        if (len(loans_to_purchase) > 0) & (available_cash > 25):
            # buy some loans!
            payload = json.dumps({'aid': account_id, 'orders': [{'loanId': list(loans_to_purchase)[count],
                                  'requestedAmount': float(25),    'portfolioId': portfolio_id}]})

            buy_loans_r = requests.post('https://api.lendingclub.com/api/investor/v1/accounts/{}/orders'.format(account_id),
                                        data=payload,
                                        headers={'Authorization': credentials['authentication'],'Content-Type': 'application/json',
                                        'Accept': 'application/json'}, params={'showAll':'true'})
            count += 1

            sys.stderr.write("Purchased one loan at {} \n".format(
                datetime.datetime.today()))
        else:
            sys.stderr.write('Nothing to purchase at {} \n'.format(
                 datetime.datetime.today()))
            break


if __name__ == "__main__":
    # Get the credentials
    credentials = yaml.load(open(os.path.expanduser('~/.ssh/lending_club_api.yml')))
    account_id = credentials['account_id']
    # - Make the api call for the portfolios owned
    #     - Find the API portfolio to put loans into
    portfolio_r = requests.get("https://api.lendingclub.com/api/investor/v1/accounts/{}/portfolios".format(account_id),
                     headers={'Authorization': credentials['authentication'],'Content-Type': 'application/json',
                              'Accept': 'application/json'}, params = {'showAll':'true'})
    portfolio_id = 0
    for portfolio in portfolio_r.json()['myPortfolios']:
        if portfolio['portfolioName'] == 'api': # Put api orders in this portfolio
            portfolio_id = portfolio['portfolioId']
    # See how much available cash we have
    available_cash = check_cash(credentials)
    if available_cash < 25:
        sys.stderr.write("Not enough cash at {}".format(
            datetime.datetime.today()))
        sys.exit()
    # get the loans on the platform
    sys.stderr.write("Retreiving loans \n")
    loans_df = get_loans(credentials)
    # munge the loans to match the format of the trained model
    final_api_df = create_feature_columns(loans_df)
    # open the saved model
    loaded_gb_model = load_model()
    # Predic the loan success
    loans_to_purchase = predict_loan_success(loaded_gb_model, final_api_df)
    sys.stderr.write("Loans to purchase = {} \n".format(loans_to_purchase.sum()))
    # purchase loans
    purchase_loans(available_cash, account_id, loans_to_purchase,
                       portfolio_id, credentials)
