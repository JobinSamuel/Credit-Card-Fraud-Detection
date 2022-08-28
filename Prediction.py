# Making individual Predictions
def make_ind_prediction(new_data):
    data = new_data.values.reshape(1, -1)
    data = robust_scaler.transform(data)
    prob = logistic_regression.predict_proba(data)[0][1]
    if prob >= 0.2:
        return 'Will default'
    else:
        return 'Will pay'
pay = default[default['default.payment.next.month']==0]
pay.head()
pay = default[default['default.payment.next.month']==1]
pay.head()
from collections import OrderedDict
new_customer = pd.Series({'limit_bal':4000, 'áge':50, 
                          'bill_amt1':500, 'bill_amt2':35509,'bill_amt3':689,'bill_amt4':0,'bill_amt5':0,'bill_amt6':0,
                                 'pay_amt1':0, 'pay_amt2':33509, 'pay_amt3':0, 'pay_amt4':0, 'pay_amt5':0, 'pay_amt6':0,
                                 'male':1, 'grad_school':0, 'university':1, 'high_school':0, 'married':1,
                                 'pay_1':-1, 'pay_2':-1, 'pay_3':-1, 'pay_4':0, 'pay_5':-1, 'pay_6':0})
new_customer = pd.DataFrame(new_customer).transpose()
make_ind_prediction(new_customer)
pay_count = no_pay_count = 0
for x in default.index[1:]:
    if make_ind_prediction(default.loc[x].drop('default.payment.next.month')) == 'Will pay':
        pay_count+=1
    else:
        no_pay_count+=1
print('pay:',pay_count)
print('Nopay:',no_pay_count)
