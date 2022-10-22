import unittest
try:
    from . gaussian_mv import *
except ImportError:
    from gaussian_mv import *

try:
    from. lambda_distribution import *
except ImportError:
    from lambda_distribution import *
import numpy as np
import pandas as pd

class GmmUtilityTest(unittest.TestCase):


    '''
        This class is used to implement unittests for PyEDW.ServerConnect class
    '''

    def setUp(self):
        '''
        Method called to prepare the test fixture. This is called immediately before calling the test method;
        other than AssertionError or SkipTest, any exception raised by this method will be considered
        an error rather than a test failure. The default implementation does nothing.
        '''
        x, y = np.random.multivariate_normal([0,0], [[1,0],[0,1]], 5000).T
        x = x.reshape(5000,1)
        y = y.reshape(5000,1)
        data = pd.DataFrame(columns=['var1'], data=x)
        data['var2'] = y
        self.gmm_util = GmmUtility(mean_vector_list = [[0,0],[0,0]], data=data, variable_names=['var1', 'var2'],
                            covariance_matrix_list = [[[1,0],[0,1]],[[1,0],[0,1]]],contribution_list=[0.5,0.5])

    def test_conditional_mean(self):
        mean,_,_,_ = self.gmm_util.conditional_moments([0,None])

        self.assertTrue(abs(mean)<0.00001)

    def test_conditional_variance(self):
        _,var,_,_ = self.gmm_util.conditional_moments([0,None])

        self.assertTrue(abs(var-1)<0.00001)

    def test_conditional_skewness(self):
        _,_,skew,_ = self.gmm_util.conditional_moments([0,None])

        self.assertTrue(skew<0.00001)

    def test_conditional_kurtosis(self):
        _,_,_,kurtosis = self.gmm_util.conditional_moments([0,None])

        self.assertTrue(abs(kurtosis-3)<0.00001)

class GeneralizedLambdaDistTest(unittest.TestCase):


    '''
        This class is used to implement unittests for PyEDW.ServerConnect class
    '''

    def setUp(self):
        '''
        Method called to prepare the test fixture. This is called immediately before calling the test method;
        other than AssertionError or SkipTest, any exception raised by this method will be considered
        an error rather than a test failure. The default implementation does nothing.
        '''
        self.gld = GeneralizedLambdaDist(0, 1, 0.1, 0.1)
        self.n_sample = 500000

    def get_difference(self,x,y):
        return abs(2*(x-y)/(x+y))

    def test_standard_normal_match(self):

        self.gld.l3 = 0.13490936
        self.gld.l4 = 0.13490936

        mean_lambda, variance_lambda, skew_lambda, kurtosis_lambda = self.gld.get_moments()

        self.assertTrue(abs(skew_lambda) <0.01 and abs(kurtosis_lambda-3) <0.01)

    def test_mean_match(self):

        self.gld.l1 = np.random.rand()*10
        self.gld.l2 = np.random.rand()*2
        self.gld.l3 = np.random.rand()*0.45
        self.gld.l4 = np.random.rand()*0.45
        mean_lambda, _, _, _ = self.gld.get_moments()

        F_vals = np.linspace(0,1,self.n_sample)
        x_vals = []
        for item in F_vals:
            x_vals.append(self.gld.quantile_function(item))

        x_vals=np.array(x_vals)

        trimming_mask = np.logical_and(x_vals>-1.2e20, x_vals<1.2e20)

        F_vals=F_vals[trimming_mask]
        x_vals=x_vals[trimming_mask]

        mean_calc = np.mean(x_vals)
        self.assertTrue(self.get_difference(mean_calc,mean_lambda) <0.01)

    def test_variance_match(self):

        self.gld.l1 = np.random.rand()*10
        self.gld.l2 = np.random.rand()*2
        self.gld.l3 = np.random.rand()*0.45
        self.gld.l4 = np.random.rand()*0.45
        _, variance_lambda, _, _ = self.gld.get_moments()

        F_vals = np.linspace(0,1,self.n_sample)
        x_vals = []
        for item in F_vals:
            x_vals.append(self.gld.quantile_function(item))

        x_vals=np.array(x_vals)

        trimming_mask = np.logical_and(x_vals>-1.2e20, x_vals<1.2e20)

        F_vals=F_vals[trimming_mask]
        x_vals=x_vals[trimming_mask]

        variance_calc = np.var(x_vals)

        self.assertTrue(self.get_difference(variance_lambda,variance_calc) <0.01)

    def test_skewness_match(self):

        self.gld.l1 = np.random.rand()*10
        self.gld.l2 = np.random.rand()*2
        self.gld.l3 = np.random.rand()*0.45
        self.gld.l4 = np.random.rand()*0.45
        _, _, skewness_lambda, _ = self.gld.get_moments()

        F_vals = np.linspace(0,1,self.n_sample)
        x_vals = []
        for item in F_vals:
            x_vals.append(self.gld.quantile_function(item))

        x_vals=np.array(x_vals)

        trimming_mask = np.logical_and(x_vals>-1.2e20, x_vals<1.2e20)

        F_vals=F_vals[trimming_mask]
        x_vals=x_vals[trimming_mask]

        mean_calc = np.mean(x_vals)
        std_calc = np.std(x_vals)
        skewness_calc = np.mean((x_vals-mean_calc)**3)/std_calc**3


        self.assertTrue(self.get_difference(skewness_lambda,skewness_calc) <0.05)

    def test_kurtosis_match(self):

        self.gld.l1 = np.random.rand()*10
        self.gld.l2 = np.random.rand()*2
        self.gld.l3 = np.random.rand()*0.45
        self.gld.l4 = np.random.rand()*0.45
        _, _, _, kurtosis_lambda = self.gld.get_moments()

        F_vals = np.linspace(0,1,self.n_sample)
        x_vals = []
        for item in F_vals:
            x_vals.append(self.gld.quantile_function(item))

        x_vals=np.array(x_vals)

        trimming_mask = np.logical_and(x_vals>-1.2e20, x_vals<1.2e20)

        F_vals=F_vals[trimming_mask]
        x_vals=x_vals[trimming_mask]

        mean_calc = np.mean(x_vals)
        std_calc = np.std(x_vals)
        kurtosis_calc = np.mean((x_vals-mean_calc)**4)/std_calc**4

        self.assertTrue(self.get_difference(kurtosis_lambda,kurtosis_calc) <0.05)


    def tearDown(self):
        '''
        Method called immediately after the test method has been called and the result recorded.
        This is called even if the test method raised an exception, so the implementation in subclasses may
        need to be particularly careful about checking internal state.
        Any exception, other than AssertionError or SkipTest, raised by this method will be
        considered an additional error rather than a test failure (thus increasing the total number of reported errors).
        This method will only be called if the setUp() succeeds, regardless of the outcome of the test method.
        The default implementation does nothing.
        '''


if __name__ == '__main__':
    unittest.main()