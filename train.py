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


def main(args):

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ts = time.time()

    # dataset = IHMDataset(
    #     root='/data/datasets/mimic3-benchmarks/data/root', train=True)
    
    dataset = IHMDataset(
        root='/data/datasets/mimic3-benchmarks/data/in-hospital-mortality', train=True)

    print("Finished dataset initializing")
    
    data_loader = DataLoader(
        dataset=dataset, batch_size=args.batch_size, shuffle=True)

    print("finished dataloader initializing")
    
    def loss_fn(recon_x, x, mean, log_var):
        loss = torch.nn.MSELoss()
        lossActual = loss(x, recon_x)

        KLD = (-0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp()) ) / x.size(0)

        return (lossActual + KLD)
        

    vae = VAE(
        encoder_layer_sizes=args.encoder_layer_sizes,
        latent_size=args.latent_size,
        decoder_layer_sizes=args.decoder_layer_sizes,
        conditional=args.conditional,
        num_labels=10 if args.conditional else 0).to(device)

    optimizer = torch.optim.Adam(vae.parameters(), lr=args.learning_rate)

    logs = defaultdict(list)
    
    print("Epochs: {:02d} Batch Size: {:02d} Learning Rate: {:.4f}".format(args.epochs, args.batch_size, args.learning_rate))
    
    total_loss = 0.0
    
    for epoch in range(args.epochs):

        tracker_epoch = defaultdict(lambda: defaultdict(dict))

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
            
            #tracker_epoch only used in graphs part
            # for i, yi in enumerate(y):
            #     id = len(tracker_epoch)
            #     tracker_epoch[id]['x'] = z[i, 0].item()
            #     tracker_epoch[id]['y'] = z[i, 1].item()
            #     tracker_epoch[id]['label'] = yi.item()

            #CALCULATE LOSS
            loss = loss_fn(recon_x, x, mean, log_var)

            #BACK PROP
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #ADDING LOSS TO LOGS
            logs['loss'].append(loss.item())

            #BELOW IS ALL PRINTING AND GRAPH MAKING
            # if iteration % args.print_every == 0 or iteration == len(data_loader)-1:
            #     print("Epoch {:02d}/{:02d} Batch {:04d}/{:d}, Loss {:9.4f}".format(
            #         epoch, args.epochs, iteration, len(data_loader)-1, loss.item()))
            
            if iteration == len(data_loader)-1:
                print("Epoch {:02d}/{:02d} Batch {:04d}/{:d}, Loss {:9.4f}".format(
                    epoch, args.epochs, iteration, len(data_loader)-1, loss.item()))

                # if args.conditional:
                #     c = torch.arange(0, 10).long().unsqueeze(1).to(device)
                #     z = torch.randn([c.size(0), args.latent_size]).to(device)
                #     x = vae.inference(z, c=c)
                # else:
                #     z = torch.randn([10, args.latent_size]).to(device)
                #     x = vae.inference(z)

#                 plt.figure()
#                 plt.figure(figsize=(5, 10))
#                 for p in range(10):
#                     plt.subplot(5, 2, p+1)
#                     if args.conditional:
#                         plt.text(
#                             0, 0, "c={:d}".format(c[p].item()), color='black',
#                             backgroundcolor='white', fontsize=8)
#                     plt.imshow(x[p].view(28, 28).cpu().data.numpy())
#                     plt.axis('off')

#                 if not os.path.exists(os.path.join(args.fig_root, str(ts))):
#                     if not(os.path.exists(os.path.join(args.fig_root))):
#                         os.mkdir(os.path.join(args.fig_root))
#                     os.mkdir(os.path.join(args.fig_root, str(ts)))

#                 plt.savefig(
#                     os.path.join(args.fig_root, str(ts),
#                                  "E{:d}I{:d}.png".format(epoch, iteration)),
#                     dpi=300)
#                 plt.clf()
#                 plt.close('all')

        # df = pd.DataFrame.from_dict(tracker_epoch, orient='index')
        # g = sns.lmplot(
        #     x='x', y='y', hue='label', data=df.groupby('label').head(100),
        #     fit_reg=False, legend=True)
        # g.savefig(os.path.join(
        #     args.fig_root, str(ts), "E{:d}-Dist.png".format(epoch)),
        #     dpi=300)


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