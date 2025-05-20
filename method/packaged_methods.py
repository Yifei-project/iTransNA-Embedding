from method.Embeddings import TransformerEmbedding, Seq2SeqEmbedding
import torch
import numpy as np 
from .utils import plot3dproj, hankel_matrix, standardize_ts, filter_components, mutual_information


def iTransNA_embedding(input, net_configs, emb_configs, lea_configs, output=None):
    if output is None:
        n_vars = input.shape[1]
        output = input
    else:
        n_vars = output.shape[1]
    assert output.shape[1] == net_configs.target_dim

    net_model = TransformerEmbedding(time_window=net_configs.seq_len, n_variables=n_vars, forward_horizon=emb_configs.L_inc,\
         n_latent=emb_configs.n_l, configs=net_configs, random_state=lea_configs.random_seed, device=lea_configs.device, enc_type='invert')
    net_model.fit(input, output, batch_num=lea_configs.batch_size, verbose=lea_configs.verbose, learning_rate=lea_configs.lr, n_epochs=lea_configs.n_epochs, \
        noise_scale=emb_configs.ε, num_NN=emb_configs.K, strength_regularizer=[emb_configs.α1, emb_configs.α2])
    
    Recon = net_model.transform(input)

    # execute PCA according to μ proportion of cumulative variances
    Recon -= Recon.mean(axis=0)
    _,_,vt = np.linalg.svd(Recon, full_matrices=0)
    new_recon = Recon @ vt.transpose()
    coords = filter_components(new_recon, p=emb_configs.μ)    
    return coords, net_model


def Transformer_embedding(input, net_configs, emb_configs, lea_configs, output=None):
    if output is None:
        n_vars = input.shape[1]
        output = input
    else:
        n_vars = output.shape[1]
    assert output.shape[1] == net_configs.target_dim

    net_model = TransformerEmbedding(time_window=net_configs.seq_len, n_variables=n_vars, forward_horizon=emb_configs.L_inc,\
         n_latent=emb_configs.n_l, configs=net_configs, random_state=lea_configs.random_seed, device=lea_configs.device, enc_type='vanilla')
    net_model.fit(input, output, batch_num=lea_configs.batch_size, verbose=lea_configs.verbose, learning_rate=lea_configs.lr, n_epochs=lea_configs.n_epochs, \
        noise_scale=emb_configs.ε, num_NN=emb_configs.K, strength_regularizer=[emb_configs.α1, emb_configs.α2])
    
    Recon = net_model.transform(input)

    # execute PCA according to μ proportion of cumulative variances
    Recon -= Recon.mean(axis=0)
    _,_,vt = np.linalg.svd(Recon, full_matrices=0)
    new_recon = Recon @ vt.transpose()
    coords = filter_components(new_recon, p=emb_configs.μ)    
    return coords, net_model


def Seq2Seq_embedding(input, net_configs, emb_configs, lea_configs, output=None):
    if output is None:
        n_vars = input.shape[1]
        output = input
    else:
        n_vars = output.shape[1]
    assert output.shape[1] == net_configs.target_dim

    net_model = Seq2SeqEmbedding(time_window=net_configs.seq_len, n_variables=n_vars, forward_horizon=emb_configs.L_inc,\
         n_latent=emb_configs.n_l, configs=net_configs, random_state=lea_configs.random_seed, device=lea_configs.device)
    net_model.fit(input, output, batch_num=lea_configs.batch_size, verbose=lea_configs.verbose, learning_rate=lea_configs.lr, n_epochs=lea_configs.n_epochs, \
        noise_scale=emb_configs.ε, num_NN=emb_configs.K, strength_regularizer=emb_configs.α2)
    
    Recon = net_model.transform(input)

    # execute PCA according to μ proportion of cumulative variances
    Recon -= Recon.mean(axis=0)
    _,_,vt = np.linalg.svd(Recon, full_matrices=0)
    new_recon = Recon @ vt.transpose()
    coords = filter_components(new_recon, p=emb_configs.μ)    
    return coords, net_model