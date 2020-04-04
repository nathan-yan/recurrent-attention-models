import torch
from torch import nn
from torch import optim 

import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt

import numpy as np

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.input_size = input_size 
        self.hidden_size = hidden_size

        # input gate, forget gate, cell gate, output gate
        self.input_hidden = nn.Linear(self.input_size, self.hidden_size * 4)
        self.hidden_hidden = nn.Linear(self.hidden_size, self.hidden_size * 4)

    def forward(self, inp, cell, hidden):
        # state -> bs x hidden * 4
        state = self.input_hidden(inp) + self.hidden_hidden(hidden)

        i, f, c, o = state.split(self.hidden_size, 1)
        input_gate = torch.sigmoid(i)
        forget_gate = torch.sigmoid(f)
        cell_gate = torch.tanh(c)
        output_gate = torch.sigmoid(o)

        cell = forget_gate * cell + input_gate * cell_gate
        hidden = output_gate * torch.tanh(cell)

        return hidden, cell
    
    def init_hidden(self, bs):
        return torch.zeros(bs, self.hidden_size), torch.zeros(bs, self.hidden_size)

class RAM(nn.Module):
    def __init__(self, hidden_size, action_size, glimpse_sizes = [12, 24, 48]):
        super().__init__()

        self.hidden_size = hidden_size
        self.action_size = action_size
        self.glimpse_sizes = glimpse_sizes

        self.glimpse_size = self.glimpse_sizes[0] ** 2 * len(self.glimpse_sizes)

        # the glimpse network forms a representation of the current glimpse with size self.hidden_size
        # this representation vector is the input to the core LSTM
        self.glimpse_location = nn.Linear(2, self.hidden_size // 2)
        self.glimpse_glimpse  = nn.Linear(self.glimpse_size, self.hidden_size // 2)
        self.glimpse_hidden = nn.Linear(self.hidden_size, self.hidden_size)

        self.core_lstm = LSTM(self.hidden_size, self.hidden_size)
        self.core_action = nn.Linear(self.hidden_size, self.action_size)
        self.core_location = nn.Linear(self.hidden_size, 2)     # these parameterize the mean of a unit-variance gaussian

    def init_hidden(self, bs):
        return self.core_lstm.init_hidden(bs)

    def forward(self, glimpse, loc, state):
        # repr -> bs x self.hidden_size / 2
        glimpse_repr = F.relu(self.glimpse_glimpse(glimpse))
        location_repr = F.relu(self.glimpse_location(loc))

        # intermediate -> bs x self.hidden_size
        intermediate = torch.cat((glimpse_repr, location_repr), -1)

        g = self.glimpse_hidden(intermediate) 

        core_hidden, cell = self.core_lstm.forward(g, *state)

        action = self.core_action(core_hidden)
        location = self.core_location(core_hidden)      # might want to use tanh? linear might be ok

        return action, location, (core_hidden, cell)

    def glimpse_sensor(self, images, locs):
        # images -> bs x size x size

        im_size = images[0].shape[0]
        # loc is a number from -1 to 1, so we need to scale them up to im_size
        scaled_locs = (locs + 1) * im_size / 2

        # glimpses are centered around a location

        glimpses = []
        for im, loc in zip(images, scaled_locs):
            loc_x, loc_y = int(loc[0]), int(loc[1])

            views = []
            for i, size in enumerate(self.glimpse_sizes):
                # left right top bottom
                l, r, t, b = loc_x - size // 2, loc_x + size // 2, loc_y - size // 2, loc_y + size // 2

                view = im[np.clip(t, 0, None):b, np.clip(l, 0, None):r]

                print(l, r, t, b)

                # pad the view, l, r, t, b
                view = F.pad(view, pad = [
                    (0 - l) * (l < 0),
                    (r - im_size) * (r > im_size),
                    (0 - t) * (t < 0),
                    (b - im_size) * (b > im_size)
                ])
            
                # view -> a x x

                # pool the image
                # view -> 1, 1, size/2, size/2
                view = F.avg_pool2d(view.view(1, 1, size, size), kernel_size = 2 ** i, stride = 2 ** i)

                # flatten each view
                view = torch.flatten(view, start_dim = 1)       # 1 x size ** 2
                views.append(view)                              
            
            glimpse = torch.cat(views, dim = -1)            # 1 x (size ** 2 * views)     
            glimpses.append(glimpse)
    
        return torch.cat(glimpses, dim = 0)                

def main():
    bs = 50
    num_glimpses = 4
    variance = 0.3

    # load dataset
    path = "./mnist_cluttered/mnist_cluttered_60.pt"

    ds = torch.load(path)
    print("Loaded MNIST cluttered dataset")

    ds = TensorDataset(ds[0], ds[1])
    dl = DataLoader(ds, batch_size = bs, shuffle = True)    # individual batches are now bs x 60 x 60

    ram = RAM(512, 10)
    optimizer = optim.Adam(ram.parameters(), lr = 0.001)
    ram.train()

    images, targets = ds[0 : 50]

    for epoch in range (20):
        for X, Y in dl:
            X = X.type(torch.FloatTensor)

            state = ram.init_hidden(bs)

            # initialize loc to 0, 0
            loc = torch.zeros(bs, 2)

            loss = 0

            for glimpse in range(num_glimpses):
                glimpses = ram.glimpse_sensor(
                    X,
                    locs = loc
                )       # glimpses -> bs x glimpse_size

                action, location, state = ram.forward(glimpses, loc, state)
                
                # pic a set of locs using fixed variance gaussian where `location` parameterizes the mean of the gaussian
                loc = location + torch.randn((bs, 2)) * variance
                print(loc)

                # the loss in this case is the pdf of loc 
                # loss -> bs
                loss += torch.sum(1/((variance * 2 * np.pi) ** 0.5) * np.e ** (-0.5 * ((loc - location) / (variance ** 0.5)) ** 2), -1)
            
            # at the end of the glimpses we take the action and check if it is correct
            # action -> bs x num_actions
            actions = torch.argmax(action, 1)
            rewards = (actions == Y) * 2 - 1            # either -1 or 1 -> bs

            loss *= rewards # -> bs
            loss = torch.mean(loss)

            # hybrid-supervised loss
            targets = torch.FloatTensor(bs, 10)
            targets.scatter(1, Y.view(-1, 1), 1.)

            loss -= torch.mean(torch.sum(F.log_softmax(action) * targets, -1))

            print(loss)

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()


    # glimpse sensor testing
    """
    plt.ion()
    plt.show()
    locs = torch.zeros(50, 2)
    c = 0

    for t in range (1000):
        c += 1 
        c %= 20

        locs[1][1] = (c - 10)/10.
        locs[1][0] = (c - 10)/10.
        
        images_ = ram.glimpse_sensor(
            images.float(),
            locs = locs
        )         

        i = 1
        plt.subplot(1, 4, 1)
        plt.cla()

        plt.imshow(images[i])

        for g in range (3):
            plt.subplot(1, 4, g + 2)
            plt.cla()

            plt.imshow(images_[i][144 * g :144 * g + 144].view(12, 12))

        plt.pause(0.01)
        
        print(c)
    """

if __name__ == "__main__":
    main()


        
