'''linear_regression.py
Subclass of Analysis that performs linear regression on data
YOUR NAME HERE
CS251 Data Analysis Visualization
Spring 2021
'''
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

import analysis


class LinearRegression(analysis.Analysis):
    '''
    Perform and store linear regression and related analyses
    '''

    def __init__(self, data):
        '''

        Parameters:
        -----------
        data: Data object. Contains all data samples and variables in a dataset.
        '''
        super().__init__(data)

        # ind_vars: Python list of strings.
        #   1+ Independent variables (predictors) entered in the regression.
        self.ind_vars = None
        # dep_var: string. Dependent variable predicted by the regression.
        self.dep_var = None

        # A: ndarray. shape=(num_data_samps, num_ind_vars)
        #   Matrix for independent (predictor) variables in linear regression
        self.A = None

        # y: ndarray. shape=(num_data_samps, 1)
        #   Vector for dependent variable predictions from linear regression
        self.y = None

        # R2: float. R^2 statistic
        self.R2 = None

        # Mean SEE. float. Measure of quality of fit
        self.m_sse = None

        # slope: ndarray. shape=(num_ind_vars, 1)
        #   Regression slope(s)
        self.slope = None
        # intercept: float. Regression intercept
        self.intercept = None
        # residuals: ndarray. shape=(num_data_samps, 1)
        #   Residuals from regression fit
        self.residuals = None

        #data
        self.data = data

        # p: int. Polynomial degree of regression model (Week 2)
        self.p = 1

    def linear_regression(self, ind_vars, dep_var):
        '''Performs a linear regression on the independent (predictor) variable(s) `ind_vars`
        and dependent variable `dep_var.

        Parameters:
        -----------
        ind_vars: Python list of strings. 1+ independent variables (predictors) entered in the regression.
            Variable names must match those used in the `self.data` object.
        dep_var: str. 1 dependent variable entered into the regression.
            Variable name must match one of those used in the `self.data` object.

        TODO:
        - Use your data object to select the variable columns associated with the independent and
        dependent variable strings.
        - Perform linear regression by using Scipy to solve the least squares problem y = Ac
        for the vector c of regression fit coefficients. Don't forget to add the coefficient column
        for the intercept!
        - Compute R^2 on the fit and the residuals.
        - By the end of this method, all instance variables should be set (see constructor).

        NOTE: Use other methods in this class where ever possible (do not write the same code twice!)
        '''
        #independent and dependent variable strings
        self.ind_vars = ind_vars
        self.dep_var = dep_var
        ind_vars = self.data.select_data(ind_vars)
        dep_var = self.data.select_data([dep_var])

        self.y = dep_var
        self.A = np.hstack([ind_vars, np.ones([ind_vars.shape[0], 1])])
        #print('this is a: ', self.A)

        c,_,_,_ = scipy.linalg.lstsq(self.A, self.y)

        #using residual, calculate and print the r**2 value
        # pred_Y = self.A @ c
        self.A = np.delete(self.A, -1, 1)

        self.slope = c[:-1]
        #print(self.slope)
        self.intercept = c[-1,-1]
        y_pred = self.predict(self.A)
        self.R2 = self.r_squared(y_pred)
        self.residuals = self.compute_residuals(y_pred)

    def predict(self, X=None):
        '''Use fitted linear regression model to predict the values of data matrix self.A.
        Generates the predictions y_pred = mA + b, where (m, b) are the model fit slope and intercept,
        A is the data matrix.

        Parameters:
        -----------
        X: ndarray. shape=(num_data_samps, num_ind_vars).
            If None, use self.A for the "x values" when making predictions.
            If not None, use X as independent var data as "x values" used in making predictions.

        Returns
        -----------
        y_pred: ndarray. shape=(num_data_samps, 1)
            Predicted y (dependent variable) values

        NOTE: You can write this method without any loops!
        '''


        if X is None:
            X = self.A
        if self.p > 1:
            #print('shape of X: ', X.shape)
            X = self.make_polynomial_matrix(X, self.p)
            #print('shape of X: ', X.shape)
        y_pred = (X @ self.slope) + self.intercept
          
    
        return y_pred

    def r_squared(self, y_pred):
        '''Computes the R^2 quality of fit statistic

        Parameters:
        -----------
        y_pred: ndarray. shape=(num_data_samps,).
            Dependent variable values predicted by the linear regression model

        Returns:
        -----------
        R2: float.
            The R^2 statistic
        '''
        #R^2
        mean_Y = np.mean(self.y)
        smd = np.sum((self.y - mean_Y)**2)
        res = np.sum((self.y - y_pred)**2)
        R2 = 1 - (res/smd)

        return R2

    def compute_residuals(self, y_pred):
        '''Determines the residual values from the linear regression model

        Parameters:
        -----------
        y_pred: ndarray. shape=(num_data_samps, 1).
            Data column for model predicted dependent variable values.

        Returns
        -----------
        residuals: ndarray. shape=(num_data_samps, 1)
            Difference between the y values and the ones predicted by the regression model at the
            data samples
        '''
        residuals = self.y - y_pred

        return residuals

    def mean_sse(self):
        '''Computes the mean sum-of-squares error in the predicted y compared the actual y values.
        See notebook for equation.

        Returns:
        -----------
        float. Mean sum-of-squares error

        Hint: Make use of self.compute_residuals
        '''
        mean = np.mean(self.compute_residuals(self.predict())**2)

        return mean

    def scatter(self, ind_var, dep_var, title):
        '''Creates a scatter plot with a regression line to visualize the model fit.
        Assumes linear regression has been already run.

        Parameters:
        -----------
        ind_var: string. Independent variable name
        dep_var: string. Dependent variable name
        title: string. Title for the plot

        TODO:
        - Use your scatter() in Analysis to handle the plotting of points. Note that it returns
        the (x, y) coordinates of the points.
        - Sample evenly spaced x values for the regression line between the min and max x data values
        - Use your regression slope, intercept, and x sample points to solve for the y values on the
        regression line.
        - Plot the line on top of the scatterplot.
        - Make sure that your plot has a title (with R^2 value in it)
        '''
        (x, y) = super().scatter(ind_var, dep_var, title)

        xdata = x.reshape(x.shape[0], 1)



        x = np.linspace(xdata[:,0].min(), xdata[:,0].max())
        
        #print('slope', self.slope)
        #print('int', self.intercept)
        # print('A', self.A.shape)
        # print('x shape: ', x.shape)
        # print('yLine shape: ', yLine.shape)
        
        

        if self.p > 1:
            #print('polynomial matrix shape: ', self.make_polynomial_matrix(xdata, self.p).shape)
            #print('x shape: ', x.shape)
            # print('A1\n', self.make_polynomial_matrix(xdata, self.p))
            x = x[:, np.newaxis]
            Ap = self.make_polynomial_matrix(x, self.p)
            #print(Ap.shape)
            #print(self.slope.shape)
            polyLine = np.squeeze(self.intercept + Ap @ self.slope)
            # polyLine = self.intercept + ((self.make_polynomial_matrix(xdata, self.p)).T @ x)
            # print(polyLine.shape)
            plt.plot(x, polyLine, 'g')
        else:
            yLine = self.intercept + self.slope * x
            plt.plot(x, yLine.reshape(yLine.shape[1], yLine.shape[0]), 'r')

        plt.title(title + ' and R2 value: ' + str(self.R2))

    def pair_plot(self, data_vars, fig_sz=(12, 12), hists_on_diag=True):
        '''Makes a pair plot with regression lines in each panel.
        There should be a len(data_vars) x len(data_vars) grid of plots, show all variable pairs
        on x and y axes.

        Parameters:
        -----------
        data_vars: Python list of strings. Variable names in self.data to include in the pair plot.
        fig_sz: tuple. len(fig_sz)=2. Width and height of the whole pair plot figure.
            This is useful to change if your pair plot looks enormous or tiny in your notebook!
        hists_on_diag: bool. If true, draw a histogram of the variable along main diagonal of
            pairplot.

        TODO:
        - Use your pair_plot() in Analysis to take care of making the grid of scatter plots.
        Note that this method returns the figure and axes array that you will need to superimpose
        the regression lines on each subplot panel.
        - In each subpanel, plot a regression line of the ind and dep variable. Follow the approach
        that you used for self.scatter. Note that here you will need to fit a new regression for
        every ind and dep variable pair.
        - Make sure that each plot has a title (with R^2 value in it)
        '''
        fig, axes = super().pair_plot(data_vars, fig_sz, hists_on_diag)

        for row in range(len(data_vars)):
            for col in range(len(data_vars)):
                #print('column of data vars: ', data_vars[col])
                self.linear_regression([data_vars[col]],data_vars[row])
                x = np.linspace(self.data.select_data([data_vars[col]])[:,0].min(), self.data.select_data([data_vars[col]])[:,0].max())
                #print('x: ', x)
                yLine = self.intercept + self.slope * x
                #print('yLine: ', yLine)
                #print('intercept: ', self.intercept)
                #print('slope: ', self.slope)
                axes[row, col].plot(x, yLine.reshape(yLine.shape[1], yLine.shape[0]), 'r')
                axes[row, col].set_title('R^2= {:.3f}'.format(self.R2))
        
                #add histogram
                if row == col and hists_on_diag == True:
                    varNum = len(data_vars)
                    axes[row, col].remove()
                    axes[row, col] = fig.add_subplot(varNum, varNum, row * varNum + col + 1)
                    axes[row, col].hist(self.data.select_data([data_vars[row]]))
                    if col < varNum - 1:
                        axes[row, col].set_xticks([])
                    else:
                        axes[row, col].set_xlabel(data_vars[row])
                    if row > 0:
                        axes[row, col].set_yticks([])
                    else:
                        axes[row, col].set_ylabel(data_vars[row])

    def make_polynomial_matrix(self, A, p):
        '''Takes an independent variable data column vector `A and transforms it into a matrix appropriate
        for a polynomial regression model of degree `p`.

        (Week 2)

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, 1)
            Independent variable data column vector x
        p: int. Degree of polynomial regression model.

        Returns:
        -----------
        ndarray. shape=(num_data_samps, p)
            Independent variable data transformed for polynomial model.
            Example: if p=10, then the model should have terms in your regression model for
            x^1, x^2, ..., x^9, x^10.

        NOTE: There should not be a intercept term ("x^0"), the linear regression solver method
        should take care of that.
        '''
        mtrx = A

        for powr in range(2,p+1):
            new_p = np.power(A, powr)
            mtrx = np.hstack((mtrx , new_p))
    
        
        return mtrx


    def poly_regression(self, ind_var, dep_var, p):
        '''Perform polynomial regression — generalizes self.linear_regression to polynomial curves
        (Week 2)
        NOTE: For single linear regression only (one independent variable only)

        Parameters:
        -----------
        ind_var: str. Independent variable entered in the single regression.
            Variable names must match those used in the `self.data` object.
        dep_var: str. Dependent variable entered into the regression.
            Variable name must match one of those used in the `self.data` object.
        p: int. Degree of polynomial regression model.
             Example: if p=10, then the model should have terms in your regression model for
             x^1, x^2, ..., x^9, x^10, and a column of homogeneous coordinates (1s).

        TODO:
        - This method should mirror the structure of self.linear_regression (compute all the same things)
        - Differences are:
            - You create a matrix based on the independent variable data matrix (self.A) with columns
            appropriate for polynomial regresssion. Do this with self.make_polynomial_matrix.
            - You set the instance variable for the polynomial regression degree (self.p)
        '''
        #independent and dependent variable strings
        self.ind_vars = ind_var
        self.dep_var = dep_var
        ind_vars = self.data.select_data(ind_var)
        dep_var = self.data.select_data([dep_var])

        self.y = dep_var
        self.p = p
        self.A = ind_vars
        Mp = self.make_polynomial_matrix(ind_vars, p)

        # add homogenous coordinate for intercep
        A1 = np.hstack([Mp, np.ones([self.A.shape[0], 1])])
        # print('this is a: ', self.A)
        #print('this is y: ', self.y)

        c,_,_,_= scipy.linalg.lstsq(A1, self.y)
        #print('c', c)

        #using residual, calculate and print the r**2 value
        # pred_Y = self.A @ c
        # self.A = np.delete(self.A, -1, 1)

        self.slope = c[:-1]
        #print(self.slope)
        self.intercept = c[-1,-1]
        y_pred = self.predict(self.A)
        self.R2 = self.r_squared(y_pred)
        self.residuals = self.compute_residuals(y_pred)


    def get_fitted_slope(self):
        '''Returns the fitted regression slope.
        (Week 2)

        Returns:
        -----------
        ndarray. shape=(num_ind_vars, 1). The fitted regression slope(s).
        '''
        return self.slope

    def get_fitted_intercept(self):
        '''Returns the fitted regression intercept.
        (Week 2)

        Returns:
        -----------
        float. The fitted regression intercept(s).
        '''
        return self.intercept

    def initialize(self, ind_vars, dep_var, slope, intercept, p):
        '''Sets fields based on parameter values.
        (Week 2)

        Parameters:
        -----------
        ind_var: str. Independent variable entered in the single regression.
            Variable names must match those used in the `self.data` object.
        dep_var: str. Dependent variable entered into the regression.
            Variable name must match one of those used in the `self.data` object.
        slope: ndarray. shape=(num_ind_vars, 1)
            Slope coefficients for the linear regression fits for each independent var
        intercept: float.
            Intercept for the linear regression fit
        p: int. Degree of polynomial regression model.

        TODO:
        - Use parameters and call methods to set all instance variables defined in constructor.
        '''
        self.ind_vars = ind_vars
        self.dep_var = dep_var
        self.slope = slope
        self.intercept = intercept
        self.p = p
        self.A = self.data.select_data(ind_vars)
        self.y = self.data.select_data([dep_var])
        y_pred = self.predict()
        self.R2 = self.r_squared(y_pred)
        self.residuals = self.compute_residuals(y_pred)

