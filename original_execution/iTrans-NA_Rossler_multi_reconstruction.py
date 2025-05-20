from method.packaged_methods import iTransNA_embedding
import torch
import numpy as np 
import h5py as h5
import matplotlib.pyplot as plt
from method.utils import plot3dproj, hankel_matrix, standardize_ts, filter_components, mutual_information

# Rossler
class net_configs:  # multivariate
    def __init__(self) -> None:
        self.seq_len = 25             # W, also equal to 'time_window' below
        self.pred_len = 20            # L_{pred}
        self.d_model = 64             # the encoder dimension
        self.n_heads = 8              # attention head number
        self.dropout = 0.0            # set 0, dropout is not applied for all experiments
        self.d_ff = None              # default value, don't alter
        self.activation="relu"        # default activation function, don't alter
        self.e_layers = 2             # encoder layer number
        self.target_dim = 2           # the channel number of measurements, equal to 'n_variables' below


class emb_configs:  # multivariate
    def __init__(self) -> None:
        self.n_l = 10  
        self.L_inc = 0
        self.K = 10
        self.ε = 0.1
        self.α1 = 1e-3
        self.α2 = 1e-4
        self.μ = 0.999


class learn_configs: # multivariate
    def __init__(self) -> None:
        self.lr = 0.01
        self.batch_size = 512
        self.n_epochs = 200
        self.random_seed = 2024
        self.device = torch.device('cuda:0')
        self.verbose = False


if __name__ == "__main__":

    for id in range(1,101):

        file_path = "./data/Rossler/rossler_no"+str(id)+".h5"
        fr = h5.File(file_path, mode="r")
        data = fr["xM"][()].transpose()

        net_conf = net_configs()
        emb_conf = emb_configs()
        lea_conf = learn_configs()

        Recon, model = iTransNA_embedding(
            input          = data[:,[0,1]],
            net_configs    = net_conf,
            emb_configs    = emb_conf,
            lea_configs    = lea_conf
        )

        # save the final reconstruction result
        h5f = h5.File('./results/iTrans-NA/Rossler-multi/rossler_multi_no'+str(id)+'.h5', 'w')
        dset = h5f.create_dataset("Embedding", Recon.shape, dtype='f')
        dset[...] = Recon
        h5f.close()   


    # iTrans
    for id in range(1,101):

        file_path = "./data/Rossler/rossler_no"+str(id)+".h5"
        fr = h5.File(file_path, mode="r")
        data = fr["xM"][()].transpose()

        net_conf = net_configs()
        emb_conf = emb_configs()
        lea_conf = learn_configs()

        emb_conf.ε, emb_conf.α2 = None, 0   # undeployed noise-amplification

        Recon, model = iTransNA_embedding(
            input          = data[:,[0,1]],
            net_configs    = net_conf,
            emb_configs    = emb_conf,
            lea_configs    = lea_conf
        )

        # save the final reconstruction result
        h5f = h5.File('./results/iTrans/Rossler-multi/rossler_multi_no'+str(id)+'.h5', 'w')
        dset = h5f.create_dataset("Embedding", Recon.shape, dtype='f')
        dset[...] = Recon
        h5f.close()

