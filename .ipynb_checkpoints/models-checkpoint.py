import torch
import torch.nn as nn

from utils import idx2onehot

class VAE(nn.Module):

    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes,
                 conditional=False, num_labels=0, hidden=32, decode_hidden=8):

        super().__init__()

        if conditional:
            assert num_labels > 0

        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list

        self.latent_size = latent_size

        self.encoder = Encoder(
            encoder_layer_sizes, latent_size, conditional, num_labels, hidden)
        self.decoder = Decoder(
            decoder_layer_sizes, latent_size, conditional, num_labels, hidden, decode_hidden)

    def forward(self, x, c=None):

        means, log_var = self.encoder(x, c)
        
        z = self.reparameterize(means, log_var)
        
        recon_x = self.decoder(z, c, x)
        
        return recon_x, means, log_var, z

    def reparameterize(self, mu, log_var):

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return mu + eps * std

    def inference(self, z, c=None):

        recon_x = self.decoder(z, c, x)

        return recon_x


class Encoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, num_labels, hidden):

        super().__init__()

        self.conditional = conditional
        # if self.conditional:
        #     layer_sizes[0] += num_labels
        
        self.lstm = nn.LSTM(input_size=layer_sizes[0], hidden_size=hidden, num_layers=1, batch_first=True)
        
        self.linear_means = nn.Linear(hidden, hidden)
        self.linear_log_var = nn.Linear(hidden, hidden)

    def forward(self, x, c=None):

        if self.conditional:
            c = idx2onehot(c, n=10)
            x = torch.cat((x, c), dim=-1)
            
        # print("X info before LSTM")
        # print(type(x))
        # print(x.shape)
        # # batch_size, 48, 76
        
        out, (ht, ct) = self.lstm(x)
        
        # print("out info after LSTM")
        # print(out.shape)
        # # torch.Size([128, 48, bs])
        
        out = ht[-1]
#         print("Checking")
#         print(ct.shape)
#         # torch.Size([1, 128, bs])
#         print(ht.shape)
#         # torch.Size([1, 128, bs])
#         print(out.shape)
#         # torch.Size([128, bs])
#         print(out[-1].shape)
        
#         print(out[-1])
        
        
        # Need to account for varying lengths of sequences, brute selecting last col does not work
        # Padded sequences are also important (?)
        
        # print("Output info after LSTM")
        # print(out.shape)
               
        means = self.linear_means(out)
        # print(type(means))
        
        log_vars = self.linear_log_var(out)
        
        # print("Means and Log info")
        # print(means.shape)
        # print(log_vars.shape)

        return means, log_vars


class Decoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, num_labels, hidden, decode_hidden):

        super().__init__()

        self.conditional = conditional
        if self.conditional:
            input_size = latent_size + num_labels
        else:
            input_size = latent_size
        # self.output_layer = nn.Linear(layer_sizes[1], layer_sizes[1] * 48)
        self.output_layer = nn.Linear(hidden, 48*hidden)
        
        self.lstm = nn.LSTM(input_size=hidden, hidden_size=decode_hidden, num_layers=1, batch_first=True)
        
        self.output_layer2 = nn.Linear(decode_hidden, 76)

    def forward(self, z, c, x):

        if self.conditional:
            c = idx2onehot(c, n=10)
            z = torch.cat((z, c), dim=-1)
            
        # print("Starting Decoder Forward")
        # print(z.shape)
        
        # Shape of Z before anything is (batch_size, hidden)
        # ([8, 12])
        
        z = self.output_layer(z)
        
        # print("Output info after First Linear Layer")
        # print(z.shape)
        # print(type(z))
        
        # print("Output info after Reshape")
        z = z.reshape( (z.size(dim=0), 48, int((z.size(dim=1) / 48))) )
        # print(z.shape)
        
        out, _ = self.lstm(z)
        
        # print("Output info after Decoder LSTM")
        # print(out.shape)
        
        out = self.output_layer2(out)
        # print("Output info after Final Linear")
        # print(out.shape)

        return out
