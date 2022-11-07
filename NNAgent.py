
import numpy as np
import sklearn
from sklearn.model_selection import validation_curve, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import  balanced_accuracy_score, f1_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from pandas.plotting import register_matplotlib_converters
import matplotlib.pyplot as plt
import warnings
from sklearn.exceptions import ConvergenceWarning
import pandas as pd
import pdb
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning


class NNAgent():
    def __init__(self, dataset=None, scorer="balanced_accuracy"):
        self.model = None
        self.modelType = "NN"
        self.dataset = dataset
        self.optimal_params = {}
        self.param0name = "hidden_layer_sizes"
        self.param1name = "alpha"
        self.scoreMethod = scorer # score type used for all scoring for this model


        # The pre-determined parameter ranges to perform grid search over (initially scouted over wider range using random search)
        # Also used for creation of Validation Curve, where the non-studied parameter is set to the optimal value
        self.param_distribution = {
            self.param0name: [40],
            self.param1name: [0.0003]
        }

        self.avgQueryTime = None
        self.avgTrainTime = None
        self.avgTrainAccFinal = None
        self.avgValAccFinal = None
        self.finalAccuracy = None

        simplefilter("ignore", category=ConvergenceWarning)

    def initModel(self, x_train, y_train):
        self.find_optimal_params(x_train, y_train)
        assert(len(self.optimal_params) > 0) # If dict is empty, this function should throw an error

        self.model = Pipeline([
                    ('normalizer', StandardScaler()),
                    ('clf', MLPClassifier(hidden_layer_sizes= self.optimal_params[self.param0name], alpha= self.optimal_params[self.param1name] , random_state=0, max_iter=1000))
                    ])
        #self.model = Pipeline([
        #            ('normalizer', StandardScaler()),
        #            ('clf', MLPClassifier(hidden_layer_sizes= 50, alpha= 0.0003, random_state=0))
        #            ])
        self.model.fit(x_train,y_train)
        self.plot_loss_curve(x_train, y_train)

    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train)


    def predict(self, x):
        return self.model.predict(x)

    def get_final_acc(self, X, y_true):
        y_pred = self.predict(X)
        if self.scoreMethod == "f1_weighted":
            self.finalAccuracy = f1_score(y_pred,y_true)
        else:
            self.finalAccuracy = balanced_accuracy_score(y_pred,y_true)

    def plot_loss_curve(self, x_train, y_train):
        model = sklearn.base.clone(self.model)        
        model.fit(x_train, y_train)

        loss = model['clf'].loss_curve_

        plt.style.use('bmh')
        plt.figure()
        plt.plot(loss, label = 'loss',color='orange')
        plt.title('{} {} ({})'.format(self.modelType, 'Loss Curve',self.dataset))
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.savefig('./graphs/nnGraphs/{}_{}_{}.png'.format(self.modelType, 'LossCurve',self.dataset))


    def save_final_params(self):
        cols = ("avgFitTime", "avgQueryTime", "avgTrainTime", "avgTrainAccFinal","avgValAccFinal", "finalAccuracy")

        final_vals = {
            "avgQueryTime": self.avgQueryTime,
            "avgTrainTime": self.avgTrainTime,
            "avgTrainAccFinal": self.avgTrainAccFinal,
            "avgValAccFinal": self.avgValAccFinal,
            "finalAccuracy": self.finalAccuracy,
            "param0": self.optimal_params[self.param0name],
            "param1": self.optimal_params[self.param1name]
        }

        final_df = pd.DataFrame(final_vals, index=[0])

        final_df.to_csv('./results/final_{}_{}_metrics.csv'.format('NN',self.dataset))

    def get_cv_results(self, x_train, y_train):
        scores = cross_validate(self.model, x_train, y_train, scoring=self.scoreMethod, return_train_score=True)
        scores_df = pd.DataFrame(scores)

        self.avgTrainTime = np.mean(scores['fit_time'])
        self.avgQueryTime = np.mean(scores['score_time'])
        self.avgTrainAccFinal = np.mean(scores['train_score'])
        self.avgValAccFinal = np.mean(scores['test_score'])

        scores_df["fit_time_avg"] = self.avgTrainTime
        scores_df["score_time_avg"] = self.avgQueryTime
        scores_df["avg_test_score"] = self.avgValAccFinal 
        scores_df.to_csv('./results/{}_{}_tuning.csv'.format('CV_metrics',self.dataset))

    def find_optimal_params(self, x_train, y_train):
        '''
        @brief: Use a hybridization of a random search (to find granular ranges) and a a grid search
        (for more percise ranges) to find the best model
        '''

        # Credit to: https://inria.github.io/scikit-learn-mooc/python_scripts/ensemble_hyperparameters.html
        # Example shows a randomized hyperparameter tuning search process
        # This is useful for doing a macroscopic study on wide parameter ranges quickly
        # From here, we can establish a more percise range of values to do a grid search over

        #search_cv = RandomizedSearchCV(
        #    MLPClassifier(), param_distributions=param_distribution, 
        #    scoring= "accuracy", n_iter=200, random_state=0, n_jobs=2,
        #)

        search_cv = GridSearchCV(
            MLPClassifier(random_state=0), self.param_distribution, scoring=self.scoreMethod, cv=5, refit=True)
        search_cv.fit(x_train,y_train)

        columns = [f"param_{name}" for name in self.param_distribution.keys()]
        columns += ["mean_test_error", "std_test_error"]
        cv_results = pd.DataFrame(search_cv.cv_results_)
        cv_results["mean_test_error"] = -cv_results["mean_test_score"]
        cv_results["std_test_error"] = cv_results["std_test_score"]
        cv_results["delta"] = cv_results["std_test_score"]
        sorted = cv_results[columns].sort_values(by="mean_test_error")
        cv_results.to_csv('./results/{}_{}_tuning.csv'.format('NN',self.dataset))

        # Deterministically select the best scoring parameters from the grid search (even if presented with the same test error)
        # If more than one set of parameters produces the same test error, this may be discussed in analysis, but we deterministically
        # take the first set to ensure consistency
        # self.optimal_params["hidden_layer_sizes"] = sorted.iloc[0][0]
        # self.optimal_params["alpha"] = sorted.iloc[0][1]

        #self.optimal_params["hidden_layer_sizes"] = sorted.iloc[0][0]
        #self.optimal_params["alpha"] = sorted.iloc[0][1]

        self.optimal_params["hidden_layer_sizes"] = search_cv.best_params_[self.param0name]
        self.optimal_params["alpha"] = search_cv.best_params_[self.param1name]

        return


    def plot_learning_timing_curve(self, x_train, y_train):

        model = sklearn.base.clone(self.model)

        train_sizes = np.linspace(0.1,1,10)
        indicies, train_scores, val_scores, fit_times, score_times = learning_curve(model, x_train, y_train, scoring=self.scoreMethod, cv=5, train_sizes=train_sizes, return_times=True)
        final_train_scores = np.mean(train_scores, axis=1)
        final_train_scores_std = np.std(train_scores, axis=1)
        final_val_scores = np.mean(val_scores, axis=1)
        final_val_scores_std = np.std(val_scores, axis=1)
        final_fit_times = np.mean(fit_times, axis=1)
        final_fit_times_std = np.std(fit_times, axis=1)
        final_score_times = np.mean(score_times, axis=1)
        final_score_times_std = np.std(score_times, axis=1)

        total_samples = x_train.shape[0]

        scores_df = pd.DataFrame()
        scores_df["final_train_scores"] = final_train_scores
        scores_df["final_train_scores_std"] = final_train_scores_std
        scores_df["final_val_scores"] = final_val_scores
        scores_df["final_train_scores_std"] = final_train_scores_std

        scores_df["final_fit_times"] = final_fit_times
        scores_df["final_fit_times_std"] = final_fit_times_std
        scores_df["final_score_times"] = final_score_times
        scores_df["final_score_times_std"] = final_score_times_std

        scores_df.to_csv('./results/{}_{}_LC_stats.csv'.format('NN',self.dataset))

        plt.style.use('bmh')
        plt.figure()
        plt.plot(train_sizes, final_train_scores, label = 'train',color='blue')
        plt.errorbar(train_sizes, final_train_scores, yerr=final_train_scores_std,color='blue', alpha=0.3)
        plt.fill_between(train_sizes, final_train_scores - final_train_scores_std, final_train_scores+final_train_scores_std, alpha=0.1, color='blue')
        plt.plot(train_sizes, final_val_scores, label = 'test', color='red')
        plt.errorbar(train_sizes, final_val_scores, yerr=final_val_scores_std,color='red', alpha=0.3)
        plt.fill_between(train_sizes, final_val_scores - final_val_scores_std, final_val_scores+final_val_scores_std, alpha=0.1, color='red')
        plt.title('{} {} ({})'.format(self.modelType, 'Learning Curve',self.dataset))
        plt.xlabel("Proportion of Training Data")
        plt.ylabel("Score ({})".format(self.scoreMethod))
        plt.legend()
        plt.savefig('./graphs/nnGraphs/{}_{}_{}.png'.format(self.modelType, 'LearningCurve',self.dataset))

        plt.figure()
        plt.plot(train_sizes, final_fit_times, label = 'fit', color="green")
        plt.errorbar(train_sizes, final_fit_times, yerr=final_fit_times_std,color='green', alpha=0.3)
        plt.fill_between(train_sizes, final_fit_times - final_fit_times_std, final_fit_times+final_fit_times_std, alpha=0.1, color='green')
        plt.plot(train_sizes, final_score_times, label = 'query', color='purple')
        plt.errorbar(train_sizes, final_score_times, yerr=final_score_times_std,color='purple', alpha=0.3)
        plt.fill_between(train_sizes, final_score_times - final_score_times_std, final_score_times+final_score_times_std, alpha=0.1, color='purple')
        plt.title('{} {} ({})'.format(self.modelType, 'Timing Curve',self.dataset))
        plt.xlabel("Proportion of Training Data")
        plt.ylabel("Time (ms)")
        plt.legend()
        plt.savefig('./graphs/nnGraphs/{}_{}_{}.png'.format(self.modelType, 'TimingCurve',self.dataset))




    def plot_validation_curve(self, x_train, y_train):
        model = sklearn.base.clone(self.model)
        train_scores, val_scores = validation_curve(model, x_train, y_train, scoring=self.scoreMethod, cv=5, param_name="clf__hidden_layer_sizes", param_range=self.param_distribution[self.param0name])
        final_train_scores = np.mean(train_scores, axis=1)
        final_train_scores_std = np.std(train_scores, axis=1)
        final_val_scores = np.mean(val_scores, axis=1)
        final_val_scores_std = np.std(val_scores, axis=1)

        plt.style.use('bmh')
        plt.figure()
        plt.plot(self.param_distribution[self.param0name], final_train_scores, label = 'train', color="cyan")
        plt.errorbar(self.param_distribution[self.param0name], final_train_scores, yerr=final_train_scores_std,color='cyan', alpha=0.3)
        plt.fill_between(self.param_distribution[self.param0name], final_train_scores - final_train_scores_std, final_train_scores+final_train_scores_std, alpha=0.1, color='cyan')
        plt.plot(self.param_distribution[self.param0name], final_val_scores, label = 'test', color='magenta')
        plt.errorbar(self.param_distribution[self.param0name], final_val_scores, yerr=final_val_scores_std,color='magenta', alpha=0.3)
        plt.fill_between(self.param_distribution[self.param0name], final_val_scores - final_val_scores_std, final_val_scores+final_val_scores_std, alpha=0.1, color='magenta')
        plt.title('{} {} ({}, {})'.format(self.modelType, 'Validation Curve', self.param0name,  self.dataset))
        plt.xlabel(self.param0name)
        plt.ylabel("Score ({})".format(self.scoreMethod))
        plt.legend()
        plt.savefig('./graphs/nnGraphs/{}_{}_{}.png'.format('VC', self.param0name, self.dataset))


        model = sklearn.base.clone(self.model)
        train_scores, val_scores = validation_curve(model, x_train, y_train, scoring=self.scoreMethod, cv=5, param_name="clf__alpha", param_range=self.param_distribution[self.param1name])
        final_train_scores = np.mean(train_scores, axis=1)
        final_train_scores_std = np.std(train_scores, axis=1)
        final_val_scores = np.mean(val_scores, axis=1)
        final_val_scores_std = np.std(val_scores, axis=1)

        plt.figure()
        plt.plot(self.param_distribution[self.param1name], final_train_scores, label = 'train', color="cyan")
        plt.errorbar(self.param_distribution[self.param1name], final_train_scores, yerr=final_train_scores_std,color='cyan', alpha=0.3)
        plt.fill_between(self.param_distribution[self.param1name], final_train_scores - final_train_scores_std, final_train_scores+final_train_scores_std, alpha=0.1, color='cyan')
        plt.plot(self.param_distribution[self.param1name], final_val_scores, label = 'test', color='magenta')
        plt.errorbar(self.param_distribution[self.param1name], final_val_scores, yerr=final_val_scores_std,color='magenta', alpha=0.3)
        plt.fill_between(self.param_distribution[self.param1name], final_val_scores - final_val_scores_std, final_val_scores+final_val_scores_std, alpha=0.1, color='magenta')
        plt.title('{} {} ({}, {})'.format(self.modelType, 'Validation Curve', self.param1name,  self.dataset))
        plt.xlabel(self.param1name)
        plt.ylabel("Score ({})".format(self.scoreMethod))
        plt.legend()
        plt.savefig('./graphs/nnGraphs/{}_{}_{}.png'.format('VC', self.param1name, self.dataset))




    def runExperiment(self):
        pass
    

        