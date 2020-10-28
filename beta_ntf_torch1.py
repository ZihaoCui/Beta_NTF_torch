# -*- coding: utf-8 -*-
"""
--------------------------------------------
modified by Zihao Cui 2018

Copyright © 2018 SASPL
Auteur(s) : ZIHAO CUI
the origin one is also given

--------------------------------------------
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
import torch
import string
import time
import numpy as np


def _betadiv(a, b, beta):
    if beta == 0:
        return a / b - torch.log(a / b) - 1
    if beta == 1:
        return a * (torch.log(a) - torch.log(b)) + b - a
    return (1. / beta / (beta - 1.) * (a ** beta + (beta - 1.)
            * b ** beta - beta * a * b ** (beta - 1)))

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
    def __init__(self, data_shape, n_components=50, beta=1, n_iter=500, sparse=0.,fixed_factors=[],
                 verbose=False, eps=1E-15):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.fixed_factors = fixed_factors
        self.data_shape = data_shape
        self.n_components = n_components
        self.beta = float(beta)
        self.n_iter = n_iter
        self.verbose = verbose
        self.eps = eps
        if isinstance(sparse,float) or isinstance(sparse,int):
            self.sparse = np.ones([len(data_shape)])*sparse
        else:
            self.sparse = sparse
        self.factors_= [self.nnrandn((dim, self.n_components)) for dim in data_shape]
        self.total_iter = 0

    def fit(self, Data, n_iter=-1, train=False):
        """Learns NTF model
        
        Parameters
        ----------
        X[0] : ndarray with nonnegative entries
               The input array
        """
        if train:
            tt = 100
        else:
            tt = 5
        if n_iter != -1:
            self.n_iter = n_iter
        self.total_iter += self.n_iter
        eps = self.eps
        beta = self.beta
        ndims = len(self.data_shape)
        if self.verbose:
            print ('Fitting NTF model with %d iterations....' % self.n_iter)
        X = torch.tensor(np.abs(Data),dtype=torch.float64).cuda()
		V = 0
        # main loop
        for it in range(self.n_iter):
			# show the runtime and loss
            if self.verbose and ((it+1)%tt==0 or it == 0):
                if 'tick' not in locals():
                    tick = time.time()
                s_s = self.score([X])
                print ('NTF model, iteration %d / %d, duration=%.1fms, cost=%5f'
                       % (it+1, self.n_iter, (time.time() - tick) * 1000, s_s))
                tick = time.time()

            #updating each factor in turn
            for dim in range(ndims):
                if dim in self.fixed_factors:
                    continue
                # get current model
                model = parafac(self.factors_)
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
                if len(X.shape) == 1:
                    operand_data_numerator = [X[:,None] * W * (model[...] ** (beta - 2.))]
                else:
                    operand_data_numerator = [X * W * (model[...] ** (beta - 2.))]
                operand_data_denominator = [W * (model[...] ** (beta - 1.))]
                # compute numerator and denominator for the update
                numerator = eps + torch.einsum(request, (operand_factors + operand_data_numerator))
                # ss_square = torch.dot(self.factors_[dim].T,self.factors_[dim])
                denominator = eps + torch.einsum(request, (operand_factors + operand_data_denominator)) + self.sparse[dim]

                # multiplicative update
                self.factors_[dim] *= numerator / denominator
				
        if self.verbose:
            print ('Done.')
        return self

    def _tocpu(self):
        if self.device == None:
            return
        self.device = None
        self.factors_ = [dim.cpu() for dim in self.factors_]
        self.W = self.W.cpu()

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
        model = torch.einsum(request, self.factors_)
        return model.__getitem__(key)

    def nnrandn(self,shape):
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
        return torch.abs(torch.randn(*shape,dtype=torch.float64,device=self.device))


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
    return torch.einsum(request, (factors[0],factors[1]))

#    print ('Shape of two_components : ',two_components.shape)
