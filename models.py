import torch
import torch.nn as nn

from utils import idx2onehot

class VAE(nn.Module):

    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes,
                 conditional=False, num_labels=0, hidden=32):

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
            decoder_layer_sizes, latent_size, conditional, num_labels, hidden)

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
        if self.conditional:
            layer_sizes[0] += num_labels
        
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
        
        out, (ht, ct) = self.lstm(x)
        
        out = ht[-1]
        
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

    def __init__(self, layer_sizes, latent_size, conditional, num_labels, hidden):

        super().__init__()

        self.conditional = conditional
        if self.conditional:
            input_size = latent_size + num_labels
        else:
            input_size = latent_size
        
        # print(layer_sizes)
        
        self.lstm = nn.LSTM(input_size=hidden, hidden_size=layer_sizes[1], num_layers=1, batch_first=True)
        
        # self.output_layer = nn.Linear(layer_sizes[1], layer_sizes[1] * 48)
        self.output_layer = nn.Linear(8*12, 8*12 * 48)

    def forward(self, z, c, x):

        if self.conditional:
            c = idx2onehot(c, n=10)
            z = torch.cat((z, c), dim=-1)
            
        print("Starting Decoder Forward")
        print(z.shape)
        
        # Shape of Z before anything is (batch_size, hidden)
        # ([8, 12])
        
        # OG way was to put it through LSTM, making it size 76, then through output, so now 76*48, then reshape to 64, 48, 76
        
        # Want to expand shape from 1 time slice to original size
        # of 64, 48, 76
        # Ryan said can be done with Fully Connected Linear layer
        # or by copying time slice to fill
        z = self.output_layer(z)
        
        # print("Output info after Fully Connected Linear Layer")
        # print(out.shape)
        # print(type(out))
        
        # print("Output info after Reshape")
        z = z.reshape(x.shape)
        # print(out.shape)
        
        out, _ = self.lstm(z)
        
        # print("Output info after Decoder LSTM")
        # print(out.shape)
        
        

        return out
