# lending_club_automation
Keras MLP model to buy Lending Club loans via the Lending Club API.
- Validation Accuracy of 77.5% with Gradient Boosting

#### Data and models stored at s3://lending-club-2017

## Steps
- 1) Dump new Lending Club csvs into data/ (NB: all need to have the same columns)
- 2) Run the scrip in 'combine_all_csvs...'
- 3) Use the Data_Cleaning.py notebook to get the Lending Club data in the correct format (you may need to pull the data model out from here)
- 4) Run the Model Fitting Script with the params for # of RF trees , # of GB trees, # of cv iterations
- 5) Update the best performing model name inside of the API_calls script
