# AnselSebastian README

Create a new environment and pip install the modules mentioned in requirements<br/>

##
Run the script "Minicomp_AnselSebastian.py"<br/>

When you run the script, it will ask you for the holdout data file (it expects a csv file)<br/>

##
Additional Store info is available at data/Store.csv<br/>


## Features used in the final model
<br/>

### Original variables:
SchoolHoliday <br>
Promo2


### Engineered variables (one-hot-encoding):<br/>
<br/>
PublicHoliday:          uses StateHoliday. <br/>
Easter:                uses StateHoliday. <br/>
Christmas:             uses StateHoliday. <br/>
storetype_a:           uses StoreType. <br/>
storetype_b:           uses StoreType. <br/>
storetype_c:           uses StoreType. <br/>
storetype_d:           uses StoreType. <br/>
assort_a:              uses Assortment. <br/>
assort_b:              uses Assortment. <br/>
assort_c:              uses Assortment. <br/>
dow_1 - dow_7:         uses DayOfWeek. <br/>
m_12:                 Dummy for December (other dummies are dropped). <br/>


### Engineered variables (mean-encoded)<br/>
Sales_avg_store:       Average Sales for each store. This variable is merged into the Store.csv and then saved as a new feature of the stores for predictions. <br/>

### Engineered variables (more complex)<br/>
DayOfWeek_recode:       just changed the sequence so that Sunday is the new 1, this can reflect the actual linear relationship between performance throughout weeks. <br/>
logDistance:           logarithm of competition distance (in order to reduce the influence of extremely high numbers). <br/>
City_center:            low competition distance (<500) and long time competition (>10 years) indicates that the shop is placed in a crucial spot. <br/>


date_delta:            0 for first day of dataset, then counts up each day until the end of dataset. Visual checking showed some stores had upwards trending Sales data(while we found no downward trends.)<br/>
monthstart:          Dummy that is one for the first couple of days of a month. Visual checking indicated higher sales.
firstdaysweek13:       Dummy identifies first and third week of month. Visual checking indicated that the first and third week of any month have higher sales average sales. <br/>
Fortnight_Days:        first day of each month is 1 and counts up to 14, then starts again at 1. <br/>


prstart :              Dummy that is 1 after the Promo2 campaigns started. <br/>
pr_campaign:          Dummy that is 1 if a campaign is running. A campaign is running if the month is mentioned in "PromoInterval" and the date is
                        after the overall start of the campaign indicated in the variables 'Promo2SinceWeek' and 'Promo2SinceYear'. <br/>
                       
Reopening:          Dummy that is 1 on a day the shop was opened , and that was preceded by at least 5 days of the shop being closed. <br/>

## Model used

XGBoost with
    'n_estimators': [1000],
    'learning_rate' : [0.1],
    'colsample_bytree': [0.7],
    'max_depth': [5],
    'reg_alpha': [1.1],
    'reg_lambda': [1.3],
    'subsample': [0.9]
.. the rest are default parameters.

We used 10-fold cross-validation.



