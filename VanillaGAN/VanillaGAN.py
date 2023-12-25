from Models import Generator, Discriminator
import torch 
import numpy as np
import torch.optim as optim

noise_vector = torch.rand(19)
real_sample = torch.rand(128)
NOISE_DIMENSION = noise_vector.shape[0]
generator = Generator(NOISE_DIMENSION, 128)
discriminator = Discrimiator(128)
EPOCHS = 100

gen_optim = optim.Adam(generator.parameters(), lr=0.0002)
disc_optim = optim.Adam(discriminator.parameters(), lr=0.0001)

def discriminatorLoss(fake, real):
    return -1 * ((1-real) + (0-fake))

def generatorLoss(fake):
    return -1 * (1 - fake)

for epoch in range(EPOCHS):
    fake_sample = Generator(noise_vector)
    fake_sample_out = discriminator(fake_sample)
    real_sample_out = discriminator(real_sample)
    genLoss = generatorLoss(fake_sample_out)
    discLoss = discriminatorLoss(fake_sample_out, real_sample_out)
    gen_optim.zero_grad()
    disc_optim.zero_grad()
    genLoss.backward()
    gen_optim.step()
    discLoss.backward()
    disc_optim.step()