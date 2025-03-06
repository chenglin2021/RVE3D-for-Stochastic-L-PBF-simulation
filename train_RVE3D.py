import wandb
import io
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset, DataLoader
import torch
import os
import torch.optim as optim
import numpy as np
import pandas as pd

import ldm.RVEnet as RVEnet

import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, training_channels, target_channels, dataloader, criterion, optimizer, device, epoch, output_folder, num_samples=1,
          use_amp=False):
    model.train()
    if use_amp:
        scaler = GradScaler()
    running_loss = 0.0

    for input_, camera_data_, sensor_data_, target_ in dataloader:
        inputs = input_[:, training_channels, :, :, :]
        camera_data = camera_data_.to(device)
        sensor_data = sensor_data_.to(device)
        targets = target_[:, target_channels, :, :, :]
        inputs[:, 0:2, :, :, 0:-1] = 0.1
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        if use_amp:
            # automatic mixed precision
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            # scales the loss, and calls backward() to create scaled gradients
            scaler.scale(loss).backward()
            # unscales gradients and calls or skips optimizer.step()
            scaler.step(optimizer)
            # updates the scale for next iteration
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)

    # Sample predictions and ground truth
    model.eval()  # Set model to evaluation mode for inference
    input_, _, _, target_ = next(iter(dataloader))
    inputs = input_[:, training_channels, :, :, :]
    targets = target_[:, target_channels, :, :, :]
    inputs[:, 0:2, :, :, 0:-1] = 0.1
    inputs, targets = inputs.to(device), targets.to(device)
    outputs = model(inputs)

    output_path = output_folder + str(epoch) + '/'
    os.makedirs(output_path, exist_ok=True)

    np.save(output_path + 'inputs_' + str(epoch) + '.npy', inputs.cpu().detach().numpy())
    np.save(output_path + 'outputs_' + str(epoch) + '.npy', outputs.cpu().detach().numpy())
    np.save(output_path + 'targets_' + str(epoch) + '.npy', targets.cpu().detach().numpy())

    for idx in range(num_samples):
        fig, axs = plt.subplots(1, 4, figsize=(10, 3))

        # Select a random index for visualization
        random_idx = torch.randint(inputs.shape[2], (1,)).item()

        # We'll use the first batch of our dataloader for these samples
        # Select the depth slice from the tensor at the random index
        # pick a channel to visualize:  ['auvel', 'avvel', 'awvel', 'csfrac', 'diff', 'fracl', 'rho', 'solfrac', 'temp']
        channel = 0
        sample_inputs = inputs[idx, channel, :, :, -1].cpu().detach().numpy()
        sample_inputs_laser = inputs[idx, -1, :, :, -1].cpu().detach().numpy()
        sample_targets = targets[idx, channel, :, :, -1].cpu().detach().numpy()
        sample_outputs = outputs[idx, channel, :, :, -1].cpu().detach().numpy()

        im1 = axs[0].imshow(sample_inputs)
        axs[0].title.set_text('Input')
        fig.colorbar(im1, ax=axs[0], fraction=0.046, pad=0.04)

        im2 = axs[1].imshow(sample_inputs_laser)
        axs[1].title.set_text('Laser')
        fig.colorbar(im2, ax=axs[1], fraction=0.046, pad=0.04)

        im3 = axs[2].imshow(sample_targets)
        axs[2].title.set_text('Ground Truth')
        fig.colorbar(im3, ax=axs[2], fraction=0.046, pad=0.04)

        im4 = axs[3].imshow(sample_outputs)
        axs[3].title.set_text('Prediction')
        fig.colorbar(im4, ax=axs[3], fraction=0.046, pad=0.04)

        plt.tight_layout()
        # plt.show()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)  # This is important, as it prevents Python from continuously using more memory when this loop is repeated

        # Convert buffer to PIL image
        buf.seek(0)
        img = Image.open(buf)

        # Log image to wandb
        wandb.log({"Predictions vs Ground Truth": wandb.Image(img)},
                  commit=False)  # use commit=False to prevent logging more than once per step

    wandb.log({"epoch_loss": epoch_loss})
    model.train()  # Set model back to training mode

    return epoch_loss


# powers = [150, 200, 250]
powers = [150]
speeds = ['0.8', '1.0', '1.2']
# speeds = ['0-8', '1-0', '1-2']
hatches = [60]
std_pos = [0.5, 0.8, 1.0]

datasets = []
for power in powers:
    for speed in speeds:
        # for hatch in hatches:
        #     dir_path = f'data/train_data_surf_WCCM/power{power}_speed{speed}_hatch{hatch}'
        for stdpos in std_pos:
            dir_path = f'data/train_data_surf_WCCM/power{power}_speed{speed}_stdpos{stdpos}'
            # dir_path = f'Y:/Data/Meso_Scale/RVE/train/power{power}_speed{speed}_hatch{hatch}'
            dataset = RVEnet.MyDatasetRAM(dir_path)
            datasets.append(dataset)

combined_dataset = ConcatDataset(datasets)
dataloader = DataLoader(combined_dataset, batch_size=1, shuffle=True, num_workers=0,  pin_memory=True)

training_channels = [3, 8, 9]
target_channels = [0, 1, 2, 3, 5, 8]

#[auvel, avvel, awvel, csfrac, diff, fracl, rho, solfrac, temp, RVE_Q, RVE_phi]
input_scaling_factors = torch.tensor([1, 1, 1, 1, 10, 1, 1, 1, 1e-3, 1, 1e-4], device='cuda')
#[auvel, avvel, awvel, csfrac, diff, fracl, rho, solfrac, temp]
output_scaling_factors = torch.tensor([1, 1, 1, 1, 1 / 10, 1, 1, 1, 1 / 1e-3], device='cuda')

input_scaling_factors = input_scaling_factors[training_channels]
output_scaling_factors = output_scaling_factors[target_channels]

weights = torch.tensor([2.0, 2.0, 0.8, 1.0, 1.0, 1e-3]).to('cuda')

model = RVEnet.RVE3D(3, 6, input_scaling_factors, output_scaling_factors).to('cuda')

criterion = RVEnet.WeightedMSELoss(weights)
optimizer = optim.Adam(model.parameters(), lr=0.0005, betas=(0.92, 0.999), eps=1e-10)

wandb.login()

output_folder = "Validation_2Dto3D/test4/"

loss_his = pd.DataFrame(columns=['epoch', 'loss'])

num_epochs = 500
lr = 0.0005
run = wandb.init(
    # Set the project where this run will be logged
    project="my-RVEnet",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": lr,
        "epochs": num_epochs,
    })

for epoch in range(num_epochs):
    loss = train(model, training_channels, target_channels, dataloader, criterion, optimizer, 'cuda', epoch, output_folder)
    print(f'Epoch {epoch+1}/{num_epochs} Loss: {loss:.4f}')
    loss_his.loc[epoch] = [epoch, loss]


wandb.finish()

#save model
torch.save(model.state_dict(), output_folder + 'model.pt')
loss_his.to_csv(output_folder + 'loss_his.csv')