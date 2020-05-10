{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\LEENDAH\\Anaconda2\\lib\\site-packages\\sklearn\\ensemble\\forest.py:246: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.\n",
      "  \"10 in version 0.20 to 100 in 0.22.\", FutureWarning)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Random Forest Mean Absolute Error:\n",
      "6862.535083333333\n",
      "Random Forest Accuracy:\n",
      "0.2400089950437303\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import csv\n",
    "# for data visualisation and statistical analysis\n",
    "import numpy as np\n",
    "from sklearn.model_selection import train_test_split\n",
    "import seaborn as sns\n",
    "sns.set_style(\"white\")\n",
    "import matplotlib.patches as mpatches\n",
    "import matplotlib.pyplot as plt\n",
    "from pylab import rcParams\n",
    "from sklearn import linear_model\n",
    "from sklearn.tree import DecisionTreeRegressor\n",
    "from sklearn.ensemble import RandomForestRegressor\n",
    "from sklearn.metrics import mean_absolute_error\n",
    "from sklearn.metrics import mean_squared_error\n",
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "%matplotlib inline \n",
    "#for warnings\n",
    "import warnings  \n",
    "# Load data and save indices of columns\n",
    "df = pd.read_csv(\"data.csv\",index_col=[0])\n",
    "# Fit and save an OneHotEncoder\n",
    "X = df[['Year_model', 'Mileage', 'Make']]\n",
    "Y = df.Price\n",
    "X = pd.get_dummies(X)\n",
    "X_train_val, X_test, y_train_val, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)\n",
    "X_train,X_val,y_train,y_val=train_test_split(X_train_val, y_train_val, test_size = 0.25, random_state = 0)\n",
    "X_train=np.array(X_train)\n",
    "X_val=np.array(X_val)\n",
    "y_val=np.array(y_val)\n",
    "X_test=np.array(X_test)\n",
    "y_test=np.array(y_test)\n",
    "\n",
    "forest_model = RandomForestRegressor()\n",
    "forest_model.fit(X_train, y_train)\n",
    "forest_prediction = forest_model.predict(X_test)\n",
    "print(forest_prediction)\n",
    "print(\"Random Forest Mean Absolute Error:\")\n",
    "print(mean_absolute_error(y_test, forest_prediction))\n",
    "print(\"Random Forest Accuracy:\")\n",
    "print(forest_model.score(X_test, y_test))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.externals import joblib\n",
    "gbr = joblib.load('model.pkl')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 2",
   "language": "python",
   "name": "python2"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 2
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython2",
   "version": "2.7.15"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
