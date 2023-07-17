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
from sklearn.cluster import KMeans

import numpy as np
from scipy import stats

from statsmodels.tsa.arima.model import ARIMA
import shap
import sklearn
from sklearn.model_selection import train_test_split

from sklearn.covariance import EmpiricalCovariance
from sklearn.covariance import MinCovDet
from scipy.stats import chi2

def load_data(config):
    print("Loading dataset")
    
    dataset_test = None
    dataset = IHMDataset(root='/data/datasets/mimic3-benchmarks/data/in-hospital-mortality', train=True, customListFile = 'train_listfile.csv')
    # dataset_test = IHMDataset(root='/data/datasets/mimic3-benchmarks/data/in-hospital-mortality', train=False, customListFile = 'test_listfile.csv')
    
    # dataset = LOSDataset(root='/data/datasets/mimic3-benchmarks/data/length-of-stay', train=True, n_samples = 1836, customListFile = 'train_listfile.csv')
    # dataset = PhenotypingDataset(root='/data/datasets/mimic3-benchmarks/data/phenotyping', train=True)

    print("Finished dataset initializing")
    
    data_loader = DataLoader(
        dataset=dataset, batch_size=config["batchSize"], shuffle=False)
    
    data_loader_test = DataLoader(
        dataset=dataset_test, batch_size=config["batchSize"], shuffle=False)

    print("finished dataloader initializing")
    
    # print(dataset[0][0].size(), dataset[1][0].size())
    
    
    return data_loader, data_loader_test

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

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    
    ts = time.time()
    
    data_loader, data_loader_test = load_data(config)
    
    vae = VAE(
        encoder_layer_sizes=args.encoder_layer_sizes,
        latent_size=args.latent_size,
        decoder_layer_sizes=args.decoder_layer_sizes,
        conditional=args.conditional,
        num_labels=2 if args.conditional else 0,
        hidden=config["hidden"],
        decode_hidden=config["decode_hidden"]).to(device)
    
    optimizer = torch.optim.Adam(vae.parameters(), lr=config["lr"])

    epochRange = config["epochs"]

    logs = defaultdict(list)
    
    pred = []
    
    reconx = None
    
    graphCount = 0
    
    for epoch in range(epochRange):

        tracker_epoch = defaultdict(lambda: defaultdict(dict))
        
        for iteration, (x, y, sl, m) in enumerate(data_loader):
            
            y = y.to(torch.int64)
            x, y = x.to(device), y.to(device)
            
            if args.conditional:
                recon_x, mean, log_var, z = vae(x, y)
            else:
                recon_x, mean, log_var, z = vae(x)
            
            reconx = recon_x
            
            # Z, mean, and log_var all have size of ([batch_size, hidden size])
            # print(z.shape)
            # print(z[0])
            
            # CHECK TO SEE IF THIS IS CORRECTLY CALCULATING AND SUBTRACTING MEANS
            df = pd.DataFrame(z.detach().numpy())
            # print(df.shape)
            # print(df.iloc[0,0])
            
            # print(df.iloc[:,0].shape)
            
            # print(np.mean(df.iloc[:,0]))
            for ind in range(df.shape[1]):
                df.iloc[:,ind] = df.iloc[:,ind] - np.mean(df.iloc[:,ind])
            diff_mean = df
            # print(diff_mean.shape)
            # print(diff_mean.iloc[0,0])
            
            covmat = np.cov(z.detach().numpy().T)
            # Transposed due to numpy covariance parameter input being inverted for rows/cols
            
            # print(covmat)
            invcovmat = np.linalg.inv(covmat)
            right = np.dot(diff_mean, invcovmat)
            mahamat = np.dot(right, diff_mean.T)
            mahalanobis = np.diag(mahamat)
            
            # print(type(mahamat))
            # print(mahamat.shape)
            # print(type(mahalanobis))
            # print(mahalanobis.shape)
            
            # ABOVE CALCULATES SQUARED MAHALANOBIS DISTANCE
            
            # cov = MinCovDet().fit(z.detach().numpy())
            # mahalanobis = cov.mahalanobis(z.detach().numpy())
            
            # print(type(mahalanobis))
            # print(mahalanobis.shape)
            # print(mahalanobis)

            pVals = 1 - chi2.cdf(mahalanobis, config["hidden"] - 1)
            # Formula for p values is correct
            
            # print(type(pVals))
            # print(pVals.shape)
            # print(pVals)
            
            significant = []
            for i, val in enumerate(pVals):
                if val <= 0.0001:
                    significant.append(i)
            # print(len(significant))
            
            if len(significant) == 128 or graphCount == 0:
                graphCount = 1
                print(type(mahalanobis))
                print(mahalanobis.shape)
                print(mahalanobis)

                print(type(pVals))
                print(pVals.shape)
                print(pVals)
             
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
                
                if (x.size(dim=0) != 1) and (not args.no_plots):
                    pca = PCA(n_components=2)
                    newZ = pca.fit_transform(z.detach().cpu().numpy())
                    # newMean = pca.fit_transform(mean.detach().cpu().numpy())
                    # newVar = pca.fit_transform(log_var.detach().cpu().numpy())
                    
                    # print(newMean.shape)
                    # newMean has shape of (8,2)
                    
                    df = pd.DataFrame(newZ)
                    
                    for ind in range(df.shape[1]):
                        df.iloc[:,ind] = df.iloc[:,ind] - np.mean(df.iloc[:,ind])
                    diff_mean = df

                    covmat = np.cov(newZ.T)
                    # Transposed due to numpy covariance parameter input being inverted for rows/cols

                    invcovmat = np.linalg.inv(covmat)
                    right = np.dot(diff_mean, invcovmat)
                    mahamat = np.dot(right, diff_mean.T)
                    mahalanobis = np.diag(mahamat)
                    
                    # cov = MinCovDet().fit(newZ)
                    # mahalanobis = cov.mahalanobis(newZ)

                    # print(type(mahalanobis))
                    # print(mahalanobis.shape)
                    # print(mahalanobis)

                    pVals = 1 - chi2.cdf(mahalanobis, 2 - 1)
                    # Formula for p values is correct

                    # print(type(pVals))
                    # print(pVals.shape)
                    # print(pVals)
                    
                    significant = []
                    for i, val in enumerate(pVals):
                        if val <= 0.001:
                            significant.append(i)
                    print(len(significant))

                    colors = [plt.cm.jet(float(i)/max(mahalanobis)) for i in mahalanobis]
                    fig = plt.figure(figsize=(8,6))
                    with plt.style.context(('ggplot')):
                        plt.scatter(newZ[:,0], newZ[:,1], c=colors, edgecolors='k', s=60)
                        plt.xlabel('x')
                        plt.ylabel('y')
                        plt.title('Outlier Color Coding')
                    fig.savefig('mahalanobisFigs/mahalanobis.png')
                    
                    for i, yi in enumerate(y):
                        id = len(tracker_epoch)
                        tracker_epoch[id]['x'] = newZ[i, 0].item()
                        tracker_epoch[id]['y'] = newZ[i, 1].item()
                        tracker_epoch[id]['label'] = yi.item()
                    # print(len(tracker_epoch))
            break
                    
        if args.tune:
            session.report(
                {"loss": loss.item()},
            )
        elif not args.no_plots:
            trackerdf = pd.DataFrame.from_dict(tracker_epoch, orient='index')
            # print(trackerdf.head(10))
            trackerdata = trackerdf.groupby('label').head(300)
            # print(trackerdata.head(10))
            trackerdata = trackerdata.drop(columns=['label'])


            kmeans = KMeans(n_clusters=2, n_init='auto')
            pred = kmeans.fit_predict(trackerdata)
            # print(pred)
        
            df = pd.DataFrame.from_dict(tracker_epoch, orient='index')
            data = df.groupby('label').head(300)
            data.insert(3, 'pred', pred)
            # print(data.head(5))
            g = sns.lmplot(
                x='x', y='y', hue='pred', data=data,
                fit_reg=False, legend=True)
            g.savefig(os.path.join(
                args.fig_root, "E{:d}-Dist.png".format(epoch)),
                dpi=300)
            
    # END OF EPOCH LOOP
    
    
    
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
    lr = 0.0005
    ep = 25
    hd = 128
    kld = 0.00001
    d_hd = 128
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    
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
    parser.add_argument("--no_plots", action='store_true')

    args = parser.parse_args()

    main(args)
