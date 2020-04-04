import argparse

import torch
import numpy as np

import matplotlib.pyplot as plt

# idea, create numbers along a long row, digits are placed randomly along the row
# attention is good for this kind of task

def main():
    sampled_digits = 1000 
    fragments_per_digit = 2
    distractions_per_digit = 4
    new_size = 60
    save_to = "./mnist_cluttered/mnist_cluttered_60.pt"
    
    # load data
    mnist = torch.load("./mnist_cluttered/MNIST/processed/training.pt")

    num_digits = mnist[0].shape[0]
    print("%s digits" % num_digits)

    # generate a group of 8 x 8 fragments from randomly selected digits
    print("Generating 8 x 8 fragments...")
    fragments = []
    for i in range (sampled_digits):
        idx = np.random.randint(0, num_digits)

        digit = mnist[0][idx]

        # get two random crops
        for crop in range (fragments_per_digit):
            x = np.random.randint(0, 20)
            y = np.random.randint(0, 20)

            fragment = digit[x : x + 8, y : y + 8]
            
            fragments.append(fragment)
        
    print("Generated %s fragments" % (sampled_digits * fragments_per_digit))

    # create new data
    mnist_cluttered = torch.ByteTensor(num_digits, new_size, new_size)

    print("Creating cluttered dataset...")
    # loop through current digits and add them to mnist_cluttered
    for i, digit in enumerate(mnist[0]):
        # add digit to mnist_cluttered with some offset
        offset_x = np.random.randint(0, new_size - 28)
        offset_y = np.random.randint(0, new_size - 28)

        mnist_cluttered[i][offset_x : offset_x + 28, offset_y : offset_y + 28] += digit

        # add fragments
        for d in range (distractions_per_digit):
            offset_x = np.random.randint(0, new_size - 8)
            offset_y = np.random.randint(0, new_size - 8)

            idx = np.random.randint(0, len(fragments))

            mnist_cluttered[i][offset_x : offset_x + 8, offset_y : offset_y + 8] += fragments[idx]

        if i % 1000 == 0:
            print("%s / %s" % (i, len(mnist[0])))

    print("Cluttered MNIST generated, saving to %s" % save_to)
    torch.save((torch.clamp(mnist_cluttered, 0, 255), mnist[1]), save_to)

if __name__ == "__main__":
    main()