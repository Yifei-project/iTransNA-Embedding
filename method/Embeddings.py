import numpy as np
import warnings
import math
import random
import os
from .utils import standardize_ts, hankel_matrix, measurement_from_state


class TimeSeriesEmbedding(object):
    # The base class of time series embedding, modified from a publicly available repository
    # see https://github.com/williamgilpin/fnn
    def __init__(self, time_window, n_latent, n_variables=1, random_state=None, **kwargs
                 ):
        
        self.time_window = time_window
        self.n_latent = n_latent
        self.n_variables = n_variables
        self.random_state = random_state
        

    def fit(self, X):
        raise AttributeError("Derived class does not contain method.")
           
    def transform(self, X):
        raise AttributeError("Derived class does not contain method.")

    def fit_transform(self, X, **kwargs):
        """Fit the model with a time series X, and then embed X.

        Parameters
        ----------
        X : array-like, shape (n_timepoints, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : None
            Ignored variable.
            
        kwargs : keyword arguments passed to the model's fit() method

        Returns
        -------
        X_new : array-like, shape (n_timepoints, n_components)
            Transformed values.
        """
        self.fit(X, **kwargs)
        return self.transform(X)   
    
    def measurement(self, X, L):    
        input = hankel_matrix(X, L).squeeze(axis=2)
        return measurement_from_state(input)

from .utils import mutual_information
from .utils import AveragedFalseNeighbors
class UnivariateUniformEmbedding(TimeSeriesEmbedding):
    """
    Embed a time series using constant (fixed) lag between values
    MI for embedding lag, AFN for embedding dimension
    ----------
    max_lag : int
    max_dim : int
    """
    def __init__(
        self, 
        # *args,
        max_lag = 10,
        max_dim = 10
        # **kwargs
    ):
        # super().__init__(*args, **kwargs)
        self.max_lag = max_lag
        self.max_dim = max_dim
        
    def calc_embed_lag(self, X):
        assert len(X.shape) == 1
        MI = mutual_information(X, self.max_lag)
        
        return MI
    
    def calc_embed_dim(self, X, tau):
        DIM = AveragedFalseNeighbors(X, self.max_dim)
        return DIM

    
    def transform(self, X, tau, d):
        # tau = self.time_window*self.lag_time
        X_test = hankel_matrix(X, q = tau*d)
        X_test = X_test[:, tau-1::tau, :]
        return np.squeeze(X_test, axis=2)
    
    def fit_transform(self, X, **kwargs):
        raise AttributeError("Derived class does not contain method.")
    
from sklearn.decomposition import PCA,  KernelPCA
class PCAEmbedding(TimeSeriesEmbedding):
    """
    PCA Embedding
    ----------
    
    kernel : "rbf" or a python function
        A nonlinear kernel to apply before performing PCA
    
    """
    def __init__(
        self, 
        *args,
        kernel=None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        if kernel:
            self.model = KernelPCA(
                n_components = self.n_latent, 
                kernel=kernel,
                random_state = self.random_state,
                copy_X=False
                )
        else:
            self.model = PCA(
                n_components = self.n_latent, 
                random_state = self.random_state
                )
                
    
    def fit(self, X):
        """Fit the model with a time series X

        Parameters
        ----------
        X : array-like, shape (n_timepoints, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : None
            Ignored variable.

        subsample : int or None
            If set to an integer, a random number of timepoints is selected
            equal to that integer

        Returns
        -------
        X_new : array-like, shape (n_timepoints, n_components)
            Transformed values.
        """        
        # Make hankel matrix from dataset
        Xs = standardize_ts(X)
        X_train = hankel_matrix(Xs, self.time_window)

        self.model.fit(np.reshape(X_train, (X_train.shape[0], -1)))
        
    def transform(self, X):
        X_test = hankel_matrix(standardize_ts(X), self.time_window)
        X_test = np.reshape(X_test, (X_test.shape[0], -1))
        X_new = self.model.transform(X_test)
        return X_new



try:     
    import torch
except Exception as e:
    warnings.warn(str(e))
    
from .Networks import RNNEncoder, RNNDecoder, TransformerEncoder, iTransformer, NoiseAmpRepresent, TransformerRepresent, Seq2SeqRepresent
from .dataset import SSRDataset, NASSRDataset
class TransformerEmbedding(TimeSeriesEmbedding):
    def __init__(
            self, *args, forward_horizon, configs, device=torch.device('cpu'), enc_type='invert', **kwargs
            ):
        super().__init__(*args, **kwargs)

        self.output_len=configs.pred_len
        self.forward_horizon=forward_horizon
        self.target_dim = configs.target_dim
        self.device=device
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            torch.cuda.manual_seed_all(self.random_state)
            np.random.seed(self.random_state)
        if enc_type == 'invert':
            enc = iTransformer(configs)
            self.model = NoiseAmpRepresent(enc, self.time_window, self.n_variables, configs.d_model, self.n_latent, output_len=self.output_len, target_dim=self.target_dim,
                                        random_state=self.random_state, device=self.device, hidden_regularizer='l1')
        if enc_type == 'vanilla':
            enc = TransformerEncoder(configs)
            self.model = TransformerRepresent(enc, self.time_window, self.n_variables, configs.d_model, self.n_latent, output_len=self.output_len, target_dim=self.target_dim,
                                        random_state=self.random_state, device=self.device, hidden_regularizer='l1')
        
    def fit(
        self,
        X,
        target,
        n_epochs=50,
        batch_num=128,
        learning_rate=1e-2,
        noise_scale=0.01,
        num_NN=3,
        clip=1,
        strength_regularizer=[0.0],
        loss_thres = 0.0,
        verbose=False,
        **kwargs
        ):
        '''
        X:      ndarray, [sample length, num_channels], multivariate time series recordings for input
        target: ndarray
        '''
        old_seed = self.random_state
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            torch.cuda.manual_seed_all(self.random_state)
            np.random.seed(self.random_state)
        dataset = NASSRDataset(standardize_ts(X), standardize_ts(target), window_size=self.time_window, \
                                forward_horizon=self.forward_horizon, output_len=self.output_len)
        batch_num = batch_num if batch_num < dataset.__len__() else dataset.__len__()
        train_iter = torch.utils.data.DataLoader(dataset, batch_size=batch_num, shuffle=True, drop_last=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
        criterion = torch.nn.functional.mse_loss
        if noise_scale is not None:
            noi_l = [noise_scale for i in range(num_NN)]   # number of nearest neighbors for one fiducial point
        else:
            noi_l = None

        Loss = []
        min = 0
        idx = 0
        for epoch in range(n_epochs):
            # if epoch > 1:
            #     weights = dynamic_weight_average(Loss[-1][:-1], Loss[-2][:-1])
            # else:
            #     weights = [1, 1]
            loss_iter = train_epoch_NA(self.model,  train_iter, optimizer, criterion, strength_regularizer, self.device, clip, \
                                       random_state=self.random_state, noise_sig=noi_l, pred_len=self.output_len)
            Loss.append(loss_iter)
            scheduler.step(sum(loss_iter))
            if epoch == 0:
                min = loss_iter
            else:
                if (min > loss_iter):
                    min = loss_iter
                    idx = epoch
            # Stop the train loop when loss has not been reduced for a period
            if epoch > (idx + 10) or min[0] < loss_thres:
                break
            if self.random_state is not None:
                self.random_state += 1
            if verbose:
                print('==== EPOCH: %d ====' %(epoch+1))
                print('Contrastive: %f  MSE: %f  Regularizer: %f' %(loss_iter[0], loss_iter[1], loss_iter[2]))
        self.random_state = old_seed
        self.loss_list = Loss

    def transform(
        self,
        X,
        **kwargs
        ):

        src = torch.from_numpy( hankel_matrix( standardize_ts(X), self.time_window ).astype(np.float32) ).to(self.device)

        self.model.eval()
        hidden = self.model.embed(src)
        hidden = hidden.cpu().numpy()
        return hidden     



class Seq2SeqEmbedding(TimeSeriesEmbedding):
    def __init__(
            self, *args, forward_horizon, target_dim=1, configs, device=torch.device('cpu'), **kwargs
            ):
        super().__init__(*args, **kwargs)

        self.output_len=configs.pred_len
        self.forward_horizon=forward_horizon
        self.target_dim=configs.target_dim
        self.enc_dim=configs.d_model
        self.n_layers=configs.e_layers
        self.device=device
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            torch.cuda.manual_seed(self.random_state)
            torch.cuda.manual_seed_all(self.random_state)
            np.random.seed(self.random_state)
            random.seed(self.random_state)
            os.environ['PYTHONHASHSEED'] =str(self.random_state)
            torch.backends.cudnn.deterministic =True
        self.build_model()

    def build_model(self):
        enc = RNNEncoder(self.n_variables, self.enc_dim, self.n_latent, n_layers=self.n_layers)
        dec = RNNDecoder(self.target_dim, self.n_latent, device=self.device)
        self.model = Seq2SeqRepresent(encoder=enc, decoder=dec, time_window=self.time_window, input_dim=self.n_variables, enc_dim=self.enc_dim, \
                                      out_dim=self.n_latent, random_state=self.random_state, device=self.device, hidden_regularizer='l1')
        
    def fit(
        self,
        input,
        target,
        n_epochs=50,
        batch_num=128,
        learning_rate=1e-2,
        clip=1,
        strength_regularizer=0.0,
        loss_thres = 0.0,
        verbose=True,
        **kwargs
        ):
        '''
        input: ndarray, [sample length, num_channels], multivariate time series recordings
        '''
        old_seed = self.random_state
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            torch.cuda.manual_seed(self.random_state)
            torch.cuda.manual_seed_all(self.random_state)
            np.random.seed(self.random_state)
            random.seed(self.random_state)
            os.environ['PYTHONHASHSEED'] =str(self.random_state)
            torch.backends.cudnn.deterministic =True
        dataset = NASSRDataset(standardize_ts(input), standardize_ts(target), window_size=self.time_window, forward_horizon=self.forward_horizon, output_len=self.output_len)
        batch_num = batch_num if batch_num < dataset.__len__() else dataset.__len__()
        train_iter = torch.utils.data.DataLoader(dataset, batch_size=batch_num, shuffle=True, drop_last=False)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.25, patience=3)
        criterion = torch.nn.functional.mse_loss

        Loss = []
        min = 0
        idx = 0
        for epoch in range(n_epochs):
            loss_iter = train_epoch(self.model, train_iter, optimizer, criterion, strength_regularizer, self.device, clip, self.random_state)
            Loss.append(loss_iter)
            scheduler.step(sum(loss_iter))
            if epoch == 0:
                min = loss_iter
            else:
                if (min > loss_iter):
                    min = loss_iter
                    idx = epoch
            # Stop the train loop when loss has not been reduced for a period
            if epoch > (idx + 10) or min[0] < loss_thres:
                break
            if self.random_state is not None:
                self.random_state += 1

            if verbose:
                print('==== EPOCH: %d ====' %(epoch+1))
                print('MSE: %f  Regularizer: %f' %(loss_iter[0], loss_iter[1]))
        self.random_state = old_seed
        self.loss_list = Loss

    def transform(
        self,
        X,
        **kwargs
        ):

        src = torch.from_numpy( hankel_matrix( standardize_ts(X), self.time_window ).astype(np.float32) ).to(self.device)

        self.model.eval()
        hidden = self.model.embed(src)
        hidden = hidden.cpu().numpy()
        return hidden     

################################################################################

# train_policy

################################################################################

def train_epoch(model, iterator, optimizer, criterion, lamb, device, clip, random_state):
    
    model.train()
    
    epoch_loss = 0
    regular_loss = 0
    if random_state is not None:
            torch.manual_seed(random_state)
            torch.cuda.manual_seed(random_state)
            torch.cuda.manual_seed_all(random_state)
            np.random.seed(random_state)
            random.seed(random_state)
            os.environ['PYTHONHASHSEED'] =str(random_state)
            torch.backends.cudnn.deterministic =True    
    for i, (src, trg) in enumerate(iterator):
        # src = [batch_size, src_len, channel_dim]
        # trg = [batch_size, trg_len, 1]
        src = src.to(device)
        trg = trg.to(device)       
        # print(src.shape, trg.shape)
        output = model(src, trg.shape[1])
        #output = [ batch size, trg_len, output dim]
        # print(output.shape)
        output_dim = output.shape[-1]
        
        output = output.view(-1, output_dim)
        trg = trg.view(-1, output_dim)
        # print(output.shape, trg.shape)
        loss1 = criterion(output, trg)
        loss2 = lamb * model.loss_reg
        loss = loss1 + loss2
        # break;
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        optimizer.zero_grad()
        
        epoch_loss += loss1.item()
        regular_loss += loss2.item()

    return [epoch_loss / len(iterator), regular_loss / len(iterator)]

def train_epoch_NA(model, iterator, optimizer, criterion, lamb, device, clip, random_state, noise_sig, pred_len):

    def loss_NA(y_true, y_pred, K, eps):
        resid = torch.max( (y_pred - y_true)**2, 3)[0]  # shape: [B, pred_len, K]
        resid = torch.mean( resid, axis=2)  # shape: [B, pred_len]
        resid = torch.sum( resid, axis=1) # shape: [B,]
        return torch.div( torch.sqrt(torch.mean(resid)), eps )

    
    model.train()
    
    epoch_loss = 0
    na_loss = 0
    K = len(noise_sig) if noise_sig is not None else 0
    if random_state is not None:
            torch.manual_seed(random_state)
            torch.cuda.manual_seed(random_state)
            torch.cuda.manual_seed_all(random_state)
            np.random.seed(random_state)
            random.seed(random_state)
            os.environ['PYTHONHASHSEED'] =str(random_state)
            torch.backends.cudnn.deterministic =True    
    for i, (src, trg) in enumerate(iterator):
        # src = [batch_size, src_len, channel_dim]
        # trg = [batch_size, trg_len, output_dim]
        src = src.to(device)
        trg = trg.to(device)   
        B = src.shape[0]    
        # print(src.shape, trg.shape)
        output, noi_o = model(src, noise_sig)
        #output = [ batch size, trg_len, output dim]
        # print(output.shape)
        if noi_o is not None:
            preds = trg[:,-1*pred_len:,:].repeat(K,1,1,1).permute(1,2,0,3)  # shape: [B,pred_len,K,output_dim]
            noi_p = noi_o[:,:,-1*pred_len:,:].permute(1,2,0,3)  # shape: [B,pred_len,K,output_dim]
            loss2 = lamb[0] * loss_NA(preds, noi_p, K, noise_sig[0])
        else:
            loss2 = torch.tensor(0.0)
        
        if type(lamb) is list and len(lamb) > 1:
            loss_reg = lamb[1] * model.loss_reg   
        else:
            loss_reg = torch.tensor(0.0)      

        output_dim = output.shape[-1]       
        output = output.view(-1, output_dim)
        trg = trg.view(-1, output_dim)
    
        loss1 = criterion(output, trg)

        loss = loss1 + loss2 + loss_reg
        # break;
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        optimizer.zero_grad()
        
        epoch_loss += loss1.item()
        na_loss += loss2.item()

    return [epoch_loss / len(iterator), na_loss / len(iterator)]


def dynamic_weight_average(loss_t_1, loss_t_2, T = 20):
    """
    :param loss_t_1: L(t-1)
    :param loss_t_2: L(t-2)
    :return: weights for each task loss
    """
    # 
    if not loss_t_1 or not loss_t_2:
        return 1

    assert len(loss_t_1) == len(loss_t_2)
    task_num = len(loss_t_1)

    w = [l_1 / l_2 for l_1, l_2 in zip(loss_t_1, loss_t_2)]

    lamb = [math.exp(v / T) for v in w]

    lamb_sum = sum(lamb)

    return [task_num * l / lamb_sum for l in lamb]