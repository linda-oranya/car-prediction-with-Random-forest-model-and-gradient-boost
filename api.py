{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " * Serving Flask app \"__main__\" (lazy loading)\n",
      " * Environment: production\n",
      "   WARNING: Do not use the development server in a production environment.\n",
      "   Use a production WSGI server instead.\n",
      " * Debug mode: off\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " * Running on http://127.0.0.1:8080/ (Press CTRL+C to quit)\n",
      "127.0.0.1 - - [09/May/2020 23:59:06] \"POST /predict?token=thisisthekey HTTP/1.1\" 401 -\n",
      "127.0.0.1 - - [09/May/2020 23:59:16] \"POST /login HTTP/1.1\" 405 -\n",
      "127.0.0.1 - - [09/May/2020 23:59:22] \"GET /login HTTP/1.1\" 200 -\n",
      "127.0.0.1 - - [09/May/2020 23:59:52] \"GET /predict?token=%22eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyIjoiQ3VyYWNlbCIsImV4cCI6MTU4OTA2NTc2Mn0.dgTw5SHIfY8NliRzbFDTYL6FLgUVONO7klvjVx84IEM%22 HTTP/1.1\" 405 -\n",
      "127.0.0.1 - - [09/May/2020 23:59:58] \"POST /predict?token=%22eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyIjoiQ3VyYWNlbCIsImV4cCI6MTU4OTA2NTc2Mn0.dgTw5SHIfY8NliRzbFDTYL6FLgUVONO7klvjVx84IEM%22 HTTP/1.1\" 401 -\n",
      "C:\\Users\\LEENDAH\\Anaconda2\\lib\\site-packages\\sklearn\\ensemble\\forest.py:246: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.\n",
      "  \"10 in version 0.20 to 100 in 0.22.\", FutureWarning)\n",
      "127.0.0.1 - - [10/May/2020 00:00:09] \"POST /predict?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyIjoiQ3VyYWNlbCIsImV4cCI6MTU4OTA2NTc2Mn0.dgTw5SHIfY8NliRzbFDTYL6FLgUVONO7klvjVx84IEM HTTP/1.1\" 200 -\n"
     ]
    }
   ],
   "source": [
    "# Dependencies\n",
    "from flask import Flask, request, jsonify\n",
    "from sklearn.externals import joblib\n",
    "import traceback\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import pickle\n",
    "import jwt,datetime\n",
    "from sklearn import linear_model\n",
    "from sklearn.tree import DecisionTreeRegressor\n",
    "from sklearn.ensemble import RandomForestRegressor\n",
    "from sklearn.metrics import mean_absolute_error\n",
    "from sklearn.metrics import mean_squared_error\n",
    "from sklearn.model_selection import train_test_split\n",
    "from datetime import datetime as dt\n",
    "from functools import wraps\n",
    "\n",
    "# Your API definition\n",
    "app = Flask(__name__)\n",
    "app.config['SECRET_KEY'] = 'thisisthekey'\n",
    "\n",
    "def token_required(f):\n",
    "    @wraps(f)\n",
    "    def decorated(*args, **kwargs):\n",
    "        token = request.args.get('token')\n",
    "        if not token:\n",
    "            return jsonify({'error':'Missing token. Set key to Token'}), 401\n",
    "        try:\n",
    "            jwt.decode(token, app.config['SECRET_KEY'])\n",
    "        except:\n",
    "            return jsonify({'error': 'Invalid token/expired'}), 401\n",
    "            \n",
    "        return f(*args, **kwargs)\n",
    "    return decorated\n",
    "\n",
    "\n",
    "@app.route('/predict', methods=['POST'])\n",
    "@token_required\n",
    "\n",
    "\n",
    "\n",
    "def predictor():   \n",
    "# Extract data in correct order\n",
    "    df = pd.read_csv(\"data.csv\",index_col=[0])\n",
    "# Fit and save an OneHotEncoder\n",
    "    X = df[['Year_model', 'Mileage', 'Make']]\n",
    "    Y = df.Price\n",
    "    X = pd.get_dummies(X)\n",
    "    X_train_val, X_test, y_train_val, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)\n",
    "    X_train,X_val,y_train,y_val=train_test_split(X_train_val, y_train_val, test_size = 0.25, random_state = 0)\n",
    "    X_train=np.array(X_train)\n",
    "    X_val=np.array(X_val)\n",
    "    y_val=np.array(y_val)\n",
    "    X_test=np.array(X_test)\n",
    "    y_test=np.array(y_test)\n",
    "    forest_model = RandomForestRegressor()\n",
    "    forest_model.fit(X_train, y_train)\n",
    "    forest_prediction = forest_model.predict(X_test)\n",
    "    return str(forest_prediction)\n",
    "\n",
    "\n",
    "\n",
    "@app.route('/login')\n",
    "def login():\n",
    "    auth = request.authorization\n",
    "    \n",
    "    if auth and auth.password == 'curacel':\n",
    "        token = jwt.encode({'user': auth.username, 'exp':dt.utcnow() + datetime.timedelta(minutes=10)},\\\n",
    "                           app.config['SECRET_KEY'])\n",
    "        return jsonify({'token':token.decode('UTF-8')})\n",
    "        \n",
    "    return make_response('could not verify!, Login is Required',401,{'WWW-Authenticate':'Basic realm = \"Login Required\"'})\n",
    "\n",
    "\n",
    "if __name__ == '__main__':\n",
    "    gbr = joblib.load(\"model.pkl\") # Load \"model.pkl\"\n",
    "    app.run(port=8080, debug=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Collecting pyjwt\n",
      "  Downloading https://files.pythonhosted.org/packages/87/8b/6a9f14b5f781697e51259d81657e6048fd31a113229cf346880bb7545565/PyJWT-1.7.1-py2.py3-none-any.whl\n",
      "Installing collected packages: pyjwt\n",
      "Successfully installed pyjwt-1.7.1\n"
     ]
    }
   ],
   "source": [
    "!pip install pyjwt\n"
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
