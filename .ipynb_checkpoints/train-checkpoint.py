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
# from tester import IHMDataset

from ray import tune
from ray.air import Checkpoint, session
from ray.tune.schedulers import ASHAScheduler

from models import VAE

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def load_data(config):
    print("Loading dataset")

    dataset = IHMDataset(root='/data/datasets/mimic3-benchmarks/data/in-hospital-mortality', train=True, customListFile = 'train_listfile.csv')
    
    # dataset = LOSDataset(root='/data/datasets/mimic3-benchmarks/data/length-of-stay', train=True, n_samples = 1836, customListFile = 'train_listfile.csv')
    # dataset = PhenotypingDataset(root='/data/datasets/mimic3-benchmarks/data/phenotyping', train=True)

    print("Finished dataset initializing")
    
    data_loader = DataLoader(
        dataset=dataset, batch_size=config["batchSize"], shuffle=True)

    print("finished dataloader initializing")
    
    return data_loader

def loss_fn(recon_x, x, mean, log_var, config):
    loss = torch.nn.MSELoss(reduction='sum')
    lossActual = loss(x, recon_x)

    KLD = config["kld"] * (-0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp()) ) 
    # Need to double check KLD calculation to see if correct, gotten from original VAE-CVAE repo

    return (lossActual + KLD) / x.size(0)

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
        num_labels=10 if args.conditional else 0,
        hidden=config["hidden"],
        decode_hidden=config["decode_hidden"]).to(device)
    
    optimizer = torch.optim.Adam(vae.parameters(), lr=config["lr"])
    # optimizer = torch.optim.RMSprop(vae.parameters(), lr=0.0003, momentum=0.1, alpha=0.001)
    
    # print("Epochs: {:02d} Batch Size: {:02d} Learning Rate: {:.4f}".format(config["epochs"], config["batchSize"], config["lr"]))
    
    epochRange = config["epochs"]

    logs = defaultdict(list)
    
    for epoch in range(epochRange):

        tracker_epoch = defaultdict(lambda: defaultdict(dict))
        
        for iteration, (x, y, sl, m) in enumerate(data_loader):
            
            x, y = x.to(device), y.to(device)

            if args.conditional:
                recon_x, mean, log_var, z = vae(x, y)
            else:
                recon_x, mean, log_var, z = vae(x)
            
#             print("Z Shape")
#             print(z.shape)
            
#             print(mean.shape)
#             print(log_var.shape)
#             print(recon_x.shape)
            
            # Z, mean, and log_var all have size of ([batch_size, hidden size])
            
            #CALCULATE LOSS
            loss = loss_fn(recon_x, x, mean, log_var, config)

            #BACK PROP
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #ADDING LOSS TO LOGS
            logs['loss'].append(loss.item())

            #BELOW IS ALL PRINTING AND GRAPH MAKING
            if not args.tune:
                if iteration == len(data_loader)-1:
                    print("Epoch {:02d} Batch {:04d}/{:d}, Loss {:9.4f}".format(
                        epoch, iteration, len(data_loader)-1, loss.item()))
                
                if x.size(dim=0) != 1:
                    pca = PCA(n_components=2)
                    newMean = pca.fit_transform(mean.detach().cpu().numpy())
                    newVar = pca.fit_transform(log_var.detach().cpu().numpy())
                    
                    # print(newMean.shape)
                    # newMean has shape of (8,2)

                    for i, yi in enumerate(y):
                        id = len(tracker_epoch)
                        tracker_epoch[id]['x'] = newMean[i, 0].item()
                        tracker_epoch[id]['y'] = newMean[i, 1].item()
                        tracker_epoch[id]['label'] = yi.item()
        
        if args.tune:
            session.report(
                {"loss": loss.item()},
            )
        else:
            df = pd.DataFrame.from_dict(tracker_epoch, orient='index')
            g = sns.lmplot(
                x='x', y='y', hue='label', data=df.groupby('label').head(300),
                fit_reg=False, legend=True)
            g.savefig(os.path.join(
                args.fig_root, "E{:d}-Dist.png".format(epoch)),
                dpi=300)
    
def main(args):
    if args.tune:
        print("Tuning!!!")
        
        # config = {
        #     "lr": tune.loguniform(1e-4, 1e-0),
        #     "batchSize": tune.grid_search([4, 8, 16, 32, 48, 64, 128]),
        #     # "epochs": tune.grid_search([5, 10, 20, 30, 40]),
        #     "epochs": args.epochs,
        #     "hidden": tune.grid_search([4, 8, 12, 24, 32, 48, 56]),
        # }
        
        config = {
            "lr": tune.grid_search([1.0, 0.1, 0.01, 0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0.000005, 0.000001, 0.0000005, 0.0000001]),
            "batchSize": args.batch_size,
            "epochs": args.epochs,
            "hidden": args.hidden,
            "kld": args.kld,
            "decode_hidden": args.decode_hidden,
        }

        trainResources = tune.with_resources(train, resources={"cpu": 1, "gpu": 1})
        tuner = tune.Tuner(
            trainable=trainResources,
            param_space=config,
            tune_config=tune.TuneConfig(num_samples=2)
        )
        
        result_grid: ResultGrid = tuner.fit()
        best: Result = result_grid.get_best_result(metric="loss", mode="min")
        
        print("Best config: ", best)
        
        print("====================================")
        
        print("Number of results: ", len(result_grid))
        print("Trial errors?: ", result_grid.errors)
        
        print("====================================")
        
        lastResult_df = result_grid.get_dataframe()
        lastResult_df = lastResult_df.sort_values(by=["loss"])
        lastResult_df = lastResult_df[["loss", "config/lr", "config/batchSize", "config/hidden", "config/epochs", "config/kld", "config/decode_hidden",]]
        print(lastResult_df.iloc[:15])
        
        bestResult_df = result_grid.get_dataframe(filter_metric="loss", filter_mode="min")
        bestResult_df = bestResult_df.sort_values(by=["loss"])
        
        lastResult_df.to_csv('lastResult.csv')
        bestResult_df.to_csv('bestResult.csv')
              
    else:
        print("Training!!!")
        
        config = {
            "lr": args.learning_rate,
            "batchSize": args.batch_size,
            "epochs": args.epochs,
            "hidden": args.hidden,
            "kld": args.kld,
            "decode_hidden": args.decode_hidden,
        }
        train(config)

if __name__ == '__main__':
    
    bs = 128
    # bs = 1
    # lr = 0.090703
    lr = 0.0005
    ep = 10
    # hd = 12
    hd = 500
    kld = 0.00001
    # d_hd = 56
    d_hd = 500
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    
    parser.add_argument("--epochs", type=int, default=ep)
    parser.add_argument("--batch_size", type=int, default=bs)
    parser.add_argument("--learning_rate", type=float, default=lr)
    parser.add_argument("--hidden", type=int, default=hd)
    parser.add_argument("--kld", type=float, default=kld)
    parser.add_argument("--decode_hidden", type=int, default=d_hd)
    
    parser.add_argument("--encoder_layer_sizes", type=list, default=[76, 32])
    parser.add_argument("--decoder_layer_sizes", type=list, default=[32, 76])
    parser.add_argument("--latent_size", type=int, default=2)
    
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--fig_root", type=str, default='newFigs')
    
    parser.add_argument("--conditional", action='store_true')
    parser.add_argument("--tune", action='store_true')

    args = parser.parse_args()

    main(args)