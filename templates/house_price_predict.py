import sys, json
print(sys.path)
import numpy as np
from sklearn.externals import joblib

data = json.loads(sys.argv[1])
data = np.array(data['data'])
sav = joblib.load('/var/www/html/house_price.sav')
pred = sav.predict(data.reshape(1,-1))
result = format(int(round(pred[0],0)),',')
print(result)
