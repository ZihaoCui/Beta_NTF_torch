# -*- coding: utf-8 -*-
"""
Copyright © 2012 Telecom ParisTech, TSI
Auteur(s) : Liutkus Antoine
the beta_ntf module is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.
You should have received a copy of the GNU LesserGeneral Public License
along with this program. If not, see <http://www.gnu.org/licenses/>."""

import string
import time
import numpy as np


def _betadiv(a, b, beta):
    if beta == 0:
        return a / b - np.log(a / b) - 1
    if beta == 1:
        return a * (np.log(a) - np.log(b)) + b - a
    return (1. / beta / (beta - 1.) * (a ** beta + (beta - 1.)
            * b ** beta - beta * a * b ** (beta - 1)))

def _yiwei(H, k):
    H = H.T
    if k>0:
        H_new = np.hstack((np.zeros([H.shape[0],k]),H[:,:H.shape[1]-k]))
    else:
        H_new = np.hstack((H[:,-k:H.shape[1]],np.zeros([H.shape[0],-k])))
    return H_new.T

class BetaNTF:
    """BetaNTF class

    Performs nonnegative parafac factorization for nonnegative ndarrays.

    This version implements:
    * Arbitrary dimension for the data to fit (actually up to 25)
    * Any beta divergence
    * Weighting of the cost function

    Parameters
    ----------
    data_shape : the shape of the data to approximate
        tuple composed of integers and of length up to 25.

    n_components : the number of latent components for the NTF model
        positive integer

    beta : the beta-divergence to consider
        Arbitrary float. Particular cases of interest are
         * beta=2 : Euclidean distance
         * beta=1 : Kullback Leibler
         * beta=0 : Itakura-Saito

    n_iter : number of iterations
        Positive integer

    fixed_factors : list of fixed factors
        list (possibly empty) of integers. if dim is in this list, factors_[dim]
        will not be updated during fit.

    Attributes
    ----------
    factors_: list of arrays
        The estimated factors
    """

    # Constructor
    def __init__(self, data_shape, n_components=50, beta=1, test=0, n_iter=500, det=0, det_value=0, sparse=0, weight=np.array([1]),cc=np.int16,
                 temporal=0, fixed_factors=[], verbose=False, eps=1E-15, W=np.array([1]),sparse_matrix=np.array([1]),gamma_p=0,
                 part_dependent=0, n_dependent=0, c_matrix=0, group=[],group_bool=np.array([1]),norms=False, cross=False,
                 part_fix=0,lamda=0):
        self.data_shape = data_shape
        self.n_components = n_components
        self.beta = float(beta)
        self.fixed_factors = fixed_factors
        self.n_iter = n_iter
        self.verbose = verbose
        self.eps = eps
        self.W = W
        self.cc = cc
        self.det_value = det_value
        self.group_bool=group_bool
        self.lamda = lamda
        if isinstance(part_fix,int):
            self.part_fix = np.ones([len(data_shape)])*part_fix
        else:
            self.part_fix = part_fix
        self.norms = norms
        if isinstance(gamma_p,int):
            self.gamma_p = np.zeros([len(data_shape)])
        else:
            self.gamma_p = gamma_p
        if isinstance(test,int):
            self.test = np.zeros([len(data_shape)])
        else:
            self.test = test
        if isinstance(c_matrix,int):
            self.c_matrix = np.zeros([len(data_shape)])
        else:
            self.c_matrix = c_matrix
        if isinstance(sparse,int):
            self.sparse = np.zeros([len(data_shape)])
        else:
            self.sparse = sparse
        if isinstance(det,int):
            self.det = np.zeros([len(data_shape)])
        else:
            self.det = det
        if isinstance(temporal,int):
            self.temporal = np.zeros([len(data_shape)])
        else:
            self.temporal = temporal
        self.factors_= [nnrandn((dim, self.n_components)) for dim in data_shape]
        self.total_iter = 0
        if isinstance(part_dependent,int):
            self.part_dependent = np.zeros([len(data_shape)])
        else:
            self.part_dependent = part_dependent
        self.n_dependent = n_dependent
        self.group = group
        self.cross = cross
        self.weight = weight
        self.sparse_matrix=sparse_matrix

        if self.gamma_p[0]:
            temp = np.sum(self.factors_[1]**2,0)
            for dim in range(2,len(data_shape)):
                temp += np.sum(self.factors_[dim]**2,0)
            self.gamma_beta = (n_components+self.gamma_p[0]-1)/(temp/2+gamma_p[1])

    def fit(self, X, n_iter=-1):
        """Learns NTF model
        
        Parameters
        ----------
        X[0] : ndarray with nonnegative entries
               The input array
        W : ndarray
            Optional ndarray that can be broadcasted with X and
            gives weights to apply on the cost function
        """
        W = self.W
        if n_iter != -1:
            self.n_iter = n_iter
        self.total_iter += self.n_iter
        eps = self.eps
        beta = self.beta
        ndims = len(self.data_shape)
        if self.verbose:
            print ('Fitting NTF model with %d iterations....' % self.n_iter)

        # main loop
        for it in range(self.n_iter):
            if self.verbose and ((it+1)%10==0 or it == 0):
                if 'tick' not in locals():
                    tick = time.time()
                s_sparse = self.score_sparse()
                s_det = self.vol_min()
                s_s = self.score([X])
                s_temporal = self.score_temporal()
                s_partdependent = self.score_kl()
                print ('NTF model, iteration %d / %d, duration=%.1fms, cost_all=%5f, cost=%5f'
                       % (it+1, self.n_iter, (time.time() - tick) * 1000,
                          s_sparse + s_det + s_s + s_temporal+s_partdependent, s_s))
                tick = time.time()

            #updating each factor in turn
            for dim in range(ndims):
                if dim in self.fixed_factors and self.part_fix[0]==self.part_fix[1]:
                    continue

                # get current model
                model = parafac(self.factors_)

                # building request for this update to use with einsum
                # for exemple for 3-way tensors, and for updating factor 2,
                # will be : 'az,cz,abc->bz'
                request = ''
                operand_factors = []
                for temp_dim in range(ndims):
                    if temp_dim == dim:
                        continue
                    request += string.ascii_lowercase[temp_dim] + 'z,'
                    operand_factors.append(self.factors_[temp_dim])
                request += string.ascii_lowercase[:ndims] + '->'
                request += string.ascii_lowercase[dim] + 'z'
                # building data-dependent factors for the update
                if len(X[0].shape) == 1:
                    operand_data_numerator = [X[0][:,None] * W * (model[...] ** (beta - 2.))]
                else:
                    operand_data_numerator = [X[0] * W * (model[...] ** (beta - 2.))]
                operand_data_denominator = [W * (model[...] ** (beta - 1.))]
                # compute numerator and denominator for the update
                numerator = eps + np.einsum(request, *(
                    operand_factors + operand_data_numerator))
                # ss_square = np.dot(self.factors_[dim].T,self.factors_[dim])
                denominator = eps + np.einsum(request, *(
                    operand_factors + operand_data_denominator)) + self.sparse[dim]*self.sparse_matrix

                if self.gamma_p[0]:
                    denominator += np.einsum('az,zz->az',self.factors_[dim],np.diag(self.gamma_beta))

                if self.test[dim]:
                    tta = self.factors_[dim][:self.factors_[dim].shape[0]-1,:]
                    ttb = self.factors_[dim][1:self.factors_[dim].shape[0],:]
                    ta = np.vstack((np.zeros([1,self.factors_[dim].shape[1]]),np.log(tta/ttb)))
                    tempt = ta
                    tempt[tempt>0] = 0
                    ta[ta<0] = 0
                    numerator += np.vstack((np.zeros([1,self.factors_[dim].shape[1]]),tta/ttb)) + tempt;
                    denominator += 1 + ta
                    #tempkk = np.vstack((np.zeros([1,self.factors_[dim].shape[1]]),_betadiv(tta, ttb, beta-1)))
                    #tempkk = tempkk/tempkk.max()*self.factors_[dim].max()
                    #denominator = denominator + self.test[dim]*tempkk
                    # print ('分母最大%d 分母最小%d' % (denominator.max(),denominator.min()))

                if self.det[dim]:
                    # geometric minimum
                    temp_ss = np.linalg.pinv(self.factors_[dim])*self.det[dim]*self.det_value
                    numerator -= (temp_ss * (temp_ss<0)).T
                    denominator += (temp_ss * (temp_ss>0)).T

                if (self.cross and it%2 ==0) or not self.cross:
                    # temporal dynamics
                    if self.temporal[dim]:
                        H = self.factors_[dim]
                        T = self.data_shape[dim]
                        sigma = np.sum(H*H,axis=0)
                        num1 = 2*T*(_yiwei(H,1)+_yiwei(H,-1))/sigma 
                        ttemp = np.sum((H-_yiwei(H,-1))*(H-_yiwei(H,-1)),axis=0)
                        num2 = (2*H*T*ttemp[np.newaxis,:])/(sigma*sigma)
                        numerator += self.temporal[dim] * (num1 + num2)
                        den = 4*T*H/sigma
                        denominator += self.temporal[dim] * den

                #uncorrelation_total
                if self.part_dependent[dim]:
                    W1 = self.factors_[dim][:,:self.n_dependent]
                    for i in range(1,self.n_dependent):
                        shift_W = np.hstack((W1[:,self.n_dependent-i:self.n_dependent],W1[:,:self.n_dependent-i]))
                        numerator[:,:self.n_dependent] += shift_W/W1 * self.part_dependent[dim] / self.n_dependent
                        denominator[:,:self.n_dependent] += self.part_dependent[dim] / self.n_dependent
                
                #similar as fixed_factors_, but not strong as it 矩阵约束
                if self.c_matrix[dim]:
                    numerator += self.c_matrix[dim]*self.group_bool[dim]*self.group[dim]/self.factors_[dim]
                    denominator += self.c_matrix[dim]*self.group_bool[dim]

                if self.lamda and dim == 0:
                    numerator += self.lamda*self.group
                    denominator += self.lamda*self.factors_[dim]

                    ##uncorrelation_part  并不能用KL散度的倒数来做
                    #if self.part_independent[dim]:
                    #    W1 = self.factors_[dim][:,:self.n_independent]
                    #    basic_shift_W = self.factors_[dim][:,self.n_independent:2*self.n_independent]
                    #    for i in range(self.n_independent):
                    #        shift_W = np.hstack((basic_shift_W[:,self.n_independent-i:self.n_independent],basic_shift_W[:,:self.n_independent-i]))
                    #        squre_temp = np.square(_betadiv(shift_W, W1, 1))+eps
                    #        numerator[:,:self.n_independent] += 1/squre_temp * self.part_independent[dim]
                    #        denominator[:,:self.n_independent] += shift_W/W1/squre_temp * self.part_independent[dim]
                    #    W1 = self.factors_[dim][:,self.n_independent:2*self.n_independent]
                    #    basic_shift_W = self.factors_[dim][:,:self.n_independent]
                    #    for i in range(self.n_independent):
                    #        shift_W = np.hstack((basic_shift_W[:,self.n_independent-i:self.n_independent],basic_shift_W[:,:self.n_independent-i]))
                    #        squre_temp = np.square(_betadiv(shift_W, W1, 1))+eps
                    #        numerator[:,self.n_independent:2*self.n_independent] += 1/squre_temp * self.part_independent[dim]
                    #        denominator[:,self.n_independent:2*self.n_independent] += shift_W/W1/squre_temp * self.part_independent[dim]
                    #print(np.min(W1))
                    #print(np.max(W1))

                # multiplicative update
                if dim in self.fixed_factors and self.part_fix[0]!=self.part_fix[1]:
                    self.factors_[dim][:,self.part_fix[0]:self.part_fix[1]] *= numerator[:,self.part_fix[0]:self.part_fix[1]] / denominator[:,self.part_fix[0]:self.part_fix[1]]
                else:
                    self.factors_[dim] *= numerator / denominator
                if dim < ndims - 1 and self.norms:
                    self.norm_fact(dim)
            if self.gamma_p[0]:
                temp = np.sum(self.factors_[1]**2,0)
                for dim in range(2,len(self.data_shape)):
                    temp += np.sum(self.factors_[dim]**2,0)
                self.gamma_beta = (self.n_components+self.gamma_p[0]-1)/(temp/2+self.gamma_p[1])
        if self.verbose:
            print ('Done.')
        return self

    def norm_fact(self,dim):
        self.factors_[dim] = (self.factors_[dim]/self.factors_[dim].sum(axis=0))
        return

    def score(self, X):
        """Computes the total beta-divergence between the current model and X

        Parameters
        ----------
        X[0] : array
            The input data

        Returns
        -------
        out : float
            The beta-divergence
        """
        return _betadiv(X[0], parafac(self.factors_), self.beta).sum()

    def score_temporal(self):
        """Computes the distance of sparse using the temporal dynamic"""
        ndim = len(self.data_shape)
        summ = 0
        for dim in range(ndim):
            if self.temporal[dim]:
                H = self.factors_[dim]
                T = self.data_shape[dim]
                sigma = np.sum(H*H,axis=0)
                summ += T*np.sum(np.sum((H-_yiwei(H,-1))*(H-_yiwei(H,-1)),axis=0)/sigma) * self.temporal[dim]
        return summ

    def vol_min(self):
        """Computes the volume of sub-matrix"""
        ndim = len(self.data_shape)
        summ = 0
        for temp in range(ndim):
            if self.det[temp]:
                summ += np.log(np.linalg.det(np.dot(self.factors_[temp].T,self.factors_[temp])*self.det[temp]*self.det_value))
        return summ

    def score_sparse(self):
        """Computes the distance of sparse using in the l1-norm"""
        ndim = len(self.data_shape)
        summ = 0
        for temp in range(ndim):
            if self.sparse[temp]:
                summ += np.sum(self.factors_[temp]) * self.sparse[temp]*self.sparse_matrix
        return summ

    def score_kl(self):
        """ Computes the KL-divergence between the shift-W and W
        Returens
        --------
        out:float
            The KL-divergence
        """
        ndim = len(self.data_shape)
        s_kl = 0
        summ = 0
        for dim in range(ndim):
            if self.part_dependent[dim]:
                    W = self.factors_[dim][:,:self.n_dependent]
                    for i in range(1,self.n_dependent):
                        shift_W = np.hstack((W[:,self.n_dependent-i:self.n_dependent],W[:,:self.n_dependent-i]))
                        summ += _betadiv(shift_W, W, 1).sum() * self.part_dependent[dim] / self.n_dependent
        return summ

    def __getitem__(self, key):
        """gets NTF model

        First compute the whole NTF model, and then call its
        __getitem__ function. Useful to get the different components. For a
        computationnaly/memory efficient approach, preferably use the
        betaNTF.parafac function

        NTF model is a ndarray of shape 
            self.data_shape+(self.n_components,)
        
        Parameters
        ----------
        key : requested slicing of the NTF model

        Returns
        -------
        ndarray containing the requested slicing of NTF model.
        
        """
        ndims = len(self.factors_)
        request = ''
        for temp_dim in range(ndims):
            request += string.ascii_lowercase[temp_dim] + 'z,'
        request = request[:-1] + '->' + string.ascii_lowercase[:ndims] + 'z'
        model = np.einsum(request, *self.factors_)
        return model.__getitem__(key)




def _totf(nparray_list):
    ndims = len(nparray_list)
    l = [None for i in range(ndims)]
    for i in range(ndims):
        l[i] = tf.convert_to_tensor(nparray_list[i])
    return l

def parafac(factors):
    """Computes the parafac model of a list of matrices

    if factors=[A,B,C,D..Z] with A,B,C..Z of shapes a*k, b*k...z*k, returns
    the a*b*..z ndarray P such that
    p(ia,ib,ic,...iz)=\sum_k A(ia,k)B(ib,k)C(ic,k)...Z(iz,k)

    Parameters
    ----------
    factors : list of arrays
        The factors

    Returns
    -------
    out : array
        The parafac model
    """
    ndims = len(factors)
    request = ''
    for temp_dim in range(ndims):
        request += string.ascii_lowercase[temp_dim] + 'z,'
    request = request[:-1] + '->' + string.ascii_lowercase[:ndims]
    return np.einsum(request, *factors)


def nnrandn(shape):
    """generates randomly a nonnegative ndarray of given shape

    Parameters
    ----------
    shape : tuple
        The shape

    Returns
    -------
    out : array of given shape
        The non-negative random numbers
    """
    return np.abs(np.random.randn(*shape))


if __name__ == '__main__':
    # Choosing the shape of the data to approximate (tuple of length up to 25)
    data_shape = (1000, 400, 10)  # 3-way tensor
    sparse = np.array([1, 0, 0])
    temporal = np.array([1, 0, 0])
    # data_shape = (1000, 400)  # matrix
    # data_shape = (50, 5, 10, 6, 7)  # 5-way tensor

    # Choosing the number of components for testing
    n_components = 9

    # Building the true factors to generate data
    factors = [nnrandn((shape, n_components)) for shape in data_shape]

    # Generating the data through the parafac function
    V = parafac(factors)

    # Create BetaNTF object with Euclidean (beta=2) distance
    beta_ntf = BetaNTF(V.shape, n_components=10, beta=1, n_iter=100,
                       verbose=True, sparse=sparse, temporal=temporal)

    # Fit the model
    beta_ntf.fit([V])

    #Print resulting score
    print ('Resulting score', beta_ntf.score(V))
    print ('Compression ratio : %0.1f%%'%((1.-sum(beta_ntf.data_shape)*
                        beta_ntf.n_components
                        /float(np.prod(beta_ntf.data_shape)))*100.))
        

    #Now illustrate the get model
    total_model= parafac(beta_ntf.factors_)
    two_components = beta_ntf[...,:2]
    print ('Shape of total_model : ',total_model.shape)
    print ('Shape of two_components : ',two_components.shape)