import os
import time
import torch
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from collections import defaultdict
from torchmimic.data import DecompensationDataset
from torchmimic.data import IHMDataset
from torchmimic.data import LOSDataset
from torchmimic.data import PhenotypingDataset

from ray import tune
from ray.air import Checkpoint, session
from ray.tune.schedulers import ASHAScheduler

from models import VAE

def load_data(config):
    # dataset = IHMDataset(
    #     root='/data/datasets/mimic3-benchmarks/data/root', train=True)
    
    dataset = IHMDataset(
        root='/data/datasets/mimic3-benchmarks/data/in-hospital-mortality', train=True)

    print("Finished dataset initializing")
    
    data_loader = DataLoader(
        dataset=dataset, batch_size=config["batchSize"], shuffle=True)

    print("finished dataloader initializing")
    
    return data_loader

def loss_fn(recon_x, x, mean, log_var):
    loss = torch.nn.MSELoss()
    lossActual = loss(x, recon_x)

    KLD = (-0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp()) ) / x.size(0)

    return (lossActual + KLD)

def train(config):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ts = time.time()
    
    data_loader = load_data(config)
    
    vae = VAE(
        encoder_layer_sizes=args.encoder_layer_sizes,
        latent_size=args.latent_size,
        decoder_layer_sizes=args.decoder_layer_sizes,
        conditional=args.conditional,
        num_labels=10 if args.conditional else 0).to(device)

    optimizer = torch.optim.Adam(vae.parameters(), lr=config["lr"])

    logs = defaultdict(list)
    
    print("Epochs: {:02d} Batch Size: {:02d} Learning Rate: {:.4f}".format(config["epochs"], config["batchSize"], config["lr"]))
    
    total_loss = 0.0
    
    for epoch in range(config["epochs"]):

        for iteration, (x, y, sl, m) in enumerate(data_loader):
            
            # print(x.shape)
            
            x, y, sl, m = x.to(device), y.to(device), sl.to(device), m.to(device)

            if args.conditional:
                recon_x, mean, log_var, z = vae(x, y)
            else:
                recon_x, mean, log_var, z = vae(x, sl)

            # print("Done with VAE")
            
            # print(recon_x.shape)
            # print(x.shape)
            
            #CALCULATE LOSS
            loss = loss_fn(recon_x, x, mean, log_var)
            
            total_loss += loss.item()

            #BACK PROP
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #ADDING LOSS TO LOGS
            logs['loss'].append(loss.item())

            #BELOW IS ALL PRINTING AND GRAPH MAKING     
            if iteration == len(data_loader)-1:
                print("Epoch {:02d} Batch {:04d}/{:d}, Loss {:9.4f}".format(
                    epoch, iteration, len(data_loader)-1, loss.item()))
                
        session.report(
            {"loss": loss.item()},
        )
        
        print("Training complete")
    
def main(args):
    config = {
        "lr": tune.loguniform(1e-4, 1e-0),
        "batchSize": tune.choice([16, 32, 64, 128, 256]),
        "epochs": tune.choice([5, 10, 15, 20, 25, 30, 35, 40]),
    }
    
    result = tune.run(
        train,
        config=config,
    )
    
    best_trial = result.get_best_trial("loss", "min", "last")
    print(f"Best config: {best_trial.config}")
    print(f"Best loss: {best_trial.last_result['loss']}")

if __name__ == '__main__':
    
    bs = 16
    lr = 0.1
    ep = 10
    # for i in range(6):
    #     bs = int(bs)
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=ep)
    parser.add_argument("--batch_size", type=int, default=bs)
    parser.add_argument("--learning_rate", type=float, default=lr)
    parser.add_argument("--encoder_layer_sizes", type=list, default=[76, 32])
    parser.add_argument("--decoder_layer_sizes", type=list, default=[32, 76])
    parser.add_argument("--latent_size", type=int, default=2)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--fig_root", type=str, default='figs')
    parser.add_argument("--conditional", action='store_true')

    args = parser.parse_args()

    main(args)
        
        # bs = bs / 2