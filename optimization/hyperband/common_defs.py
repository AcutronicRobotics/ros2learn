from __future__ import print_function
"""imports and definitions shared by various defs files"""

import numpy as np

from math import log, sqrt
from time import time
from pprint import pprint

from sklearn.metrics import roc_auc_score as AUC, log_loss, accuracy_score as accuracy
from sklearn.metrics import mean_squared_error as MSE, mean_absolute_error as MAE

try:
	from hyperopt import hp
	from hyperopt.pyll.stochastic import sample
except ImportError:
	print("In order to achieve operational capability, this programme requires hyperopt to be installed (pip install hyperopt), unless you make get_params() use something else.")

#

# handle floats which should be integers
# works with flat params
def handle_integers( params ):

	new_params = {}
	for k, v in params.items():
		if type( v ) == float and int( v ) == v:
			new_params[k] = int( v )
		else:
			new_params[k] = v

	return new_params

###

def train_and_eval_sklearn_classifier( clf, data ):

	x_train = data['x_train']
	y_train = data['y_train']

	x_test = data['x_test']
	y_test = data['y_test']

	clf.fit( x_train, y_train )

	try:
		p = clf.predict_proba( x_train )[:,1]	# sklearn convention
	except IndexError:
		p = clf.predict_proba( x_train )

	ll = log_loss( y_train, p )
	auc = AUC( y_train, p )
	acc = accuracy( y_train, np.round( p ))

	print("\n# training | log loss: {:.2%}, AUC: {:.2%}, accuracy: {:.2%}".format( ll, auc, acc ))

	#

	try:
		p = clf.predict_proba( x_test )[:,1]	# sklearn convention
	except IndexError:
		p = clf.predict_proba( x_test )

	ll = log_loss( y_test, p )
	auc = AUC( y_test, p )
	acc = accuracy( y_test, np.round( p ))

	print("# testing  | log loss: {:.2%}, AUC: {:.2%}, accuracy: {:.2%}".format( ll, auc, acc ))

	#return { 'loss': 1 - auc, 'log_loss': ll, 'auc': auc }
	return { 'loss': ll, 'log_loss': ll, 'auc': auc }

###

# "clf", even though it's a regressor
def train_and_eval_sklearn_regressor( clf, data ):

	x_train = data['x_train']
	y_train = data['y_train']

	x_test = data['x_test']
	y_test = data['y_test']

	clf.fit( x_train, y_train )
	p = clf.predict( x_train )

	mse = MSE( y_train, p )
	rmse = sqrt( mse )
	mae = MAE( y_train, p )


	print("\n# training | RMSE: {:.4f}, MAE: {:.4f}".format( rmse, mae ))

	#

	p = clf.predict( x_test )

	mse = MSE( y_test, p )
	rmse = sqrt( mse )
	mae = MAE( y_test, p )

	print("# testing  | RMSE: {:.4f}, MAE: {:.4f}".format( rmse, mae ))

	return { 'loss': rmse, 'rmse': rmse, 'mae': mae }
