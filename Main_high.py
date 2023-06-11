import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch.optim as optim
import torchvision.transforms as transforms
import os
import torch.nn.functional as F
import torch.cuda as cuda
device = torch.device("cuda" if cuda.is_available() else "cpu")




class CoarseSRNet(nn.Module):
    def __init__(self):
        super(CoarseSRNet, self).__init__()
        self.a=0
        self.enc_conv1 = nn.Conv2d(3, 256, kernel_size=3, padding=1)
        self.enc_prelu1 = nn.PReLU()
        self.enc_conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.enc_prelu2 = nn.PReLU()
        self.residual_blocks = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128)
        )
        self.dec_tconv1 = nn.ConvTranspose2d(128, 3, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_conv1 = nn.Conv2d(3, 128, kernel_size=3, padding=1)
        self.dec_tconv2 = nn.ConvTranspose2d(128, 3, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_conv2 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.dec_tconv3 = nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x = self.enc_prelu1(self.enc_conv1(x))
        x = self.enc_prelu2(self.enc_conv2(x))
        x = self.residual_blocks(x)
        x = self.dec_tconv1(x)
        x = self.dec_conv1(x)
        x = self.dec_tconv2(x)
        x = self.dec_conv2(x)
        x = self.dec_tconv3(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.prelu1 = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.prelu2 = nn.PReLU()
    def forward(self, x):
        identity = x
        x = self.prelu1(self.conv1(x))
        x = self.conv2(x)
        x += identity
        x = self.prelu2(x)
        return x

class HyFA2U(nn.Module):
    def __init__(self):
        super(HyFA2U, self).__init__()
        self.hyfeat = HyFeat()
        self.a2b=A2B()
        self.con=nn.Conv2d(124, 3, kernel_size=3, padding=1)

    def forward(self, x):
        hyfeat_output = self.hyfeat(x)
        hyfa2u_output = self.a2b(hyfeat_output)
        out=torch.cat((x,hyfa2u_output), dim=1)
        out=self.con(hyfa2u_output)
        return out

class HyFeat(nn.Module):
    def __init__(self):
        super(HyFeat, self).__init__()
        self.conv_3x3 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv_7x7 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, padding=3)
        self.conv_1x1 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, padding=0)
        self.conv_5x5 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, padding=2)

    def forward(self, x):
        conv_3x3_output = self.conv_3x3(x)
        conv_7x7_output = self.conv_7x7(x)
        concat_1 = torch.cat((conv_3x3_output, conv_7x7_output), dim=1)


        conv_1x1_output = self.conv_1x1(concat_1)
        conv_5x5_output = self.conv_5x5(concat_1)
        concat_2 = torch.cat((conv_1x1_output, conv_5x5_output), dim=1)
        return concat_2

class A2B(nn.Module):
    def __init__(self):
        super(A2B, self).__init__()
        self.Attention=Attention()
        self.Dynamic_weight=Dynamic_weight()
        self.conv = nn.Conv2d(in_channels=64, out_channels=60, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.conv(x)
        x2= self.Attention(x)
        x3=self.Dynamic_weight(x)
        x1=x1*x3
        x2=x2*x3
        x4=torch.cat((x1,x2), dim=1)
        return x4

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.prelu1 = nn.PReLU(num_parameters=64)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.deconv1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.prelu2 = nn.PReLU(num_parameters=64)
        
        self.output = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x1 = self.prelu1(self.conv1(x))
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.deconv1(x4)
        x5=torch.cat((x3,x5), dim=1)
        x6 = self.deconv2(x5)
        x6=torch.cat((x6,x2), dim=1)
        x7=self.deconv3(x6)
        x7=torch.cat((x7,x1), dim=1)
        x8 = self.prelu2(self.conv5(x7))
        x9 = self.sigmoid(self.output(x8))
        x9=torch.cat((x,x9), dim=1)
        return x8
    


class Dynamic_weight(nn.Module):
    def __init__(self):
        super(Dynamic_weight, self).__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.fc1 = nn.Linear(64*88*88, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 4*88*88)
        self.Softmax = nn.Softmax(dim=1)

    def forward(self, x):
        
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.Softmax(x)
        x = x.view(-1, 1, 176, 176)
        return x
         


class DeepCNN(nn.Module):
    def __init__(self):
        super(DeepCNN, self).__init__()
        self.hyfa2u_1 = HyFA2U()
        self.hyfa2u_2 = HyFA2U()
        self.hyfa2u_3 = HyFA2U()
        self.conv_final = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1)
        self.conv_finalx = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=3, padding=1)

    def forward(self, x):
        hyfa2u_1_output = self.hyfa2u_1(x)
        hyfa2u_2_output = self.hyfa2u_2(hyfa2u_1_output)
        hyfa2u_3_output = self.hyfa2u_3(hyfa2u_2_output)
        conv_final_output = self.conv_final(hyfa2u_3_output)
        out_x=torch.cat((conv_final_output,x), dim=1)
        out_x=self.conv_finalx(out_x)
        return out_x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)

        self.conv4 = nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.LeakyReLU(0.2, inplace=True)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(256)
        self.relu5 = nn.LeakyReLU(0.2, inplace=True)

        self.conv6 = nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(256)
        self.relu6 = nn.LeakyReLU(0.2, inplace=True)

        self.conv7 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(512)
        self.relu7 = nn.LeakyReLU(0.2, inplace=True)

        self.conv8 = nn.Conv2d(512, 1, kernel_size=2, stride=1, padding=0, bias=False)
        self.flat=nn.Flatten()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu3(self.bn3(self.conv3(out)))
        out = self.relu4(self.bn4(self.conv4(out)))
        out = self.relu5(self.bn5(self.conv5(out)))
        out = self.relu6(self.bn6(self.conv6(out)))
        out = self.relu7(self.bn7(self.conv7(out)))
        out = self.sigmoid(self.flat(self.conv8(out)))
        return out


class SuperResolutionDataset(Dataset):
    def __init__(self, low_folder, high_folder):
        self.low_images = os.listdir(low_folder)
        self.high_images = os.listdir(high_folder)
        self.low_folder = low_folder
        self.high_folder = high_folder

    def __getitem__(self, index):
        low_path = os.path.join(self.low_folder, self.low_images[index])
        high_path = os.path.join(self.high_folder, self.high_images[index])

        low_image = Image.open(low_path)
        high_image = Image.open(high_path)

        transform = transforms.ToTensor()

        low_image = transform(low_image)
        high_image = transform(high_image)

        return low_image, high_image

    def __len__(self):
        return len(self.low_images)




def psnr(target, prediction):
    target = torch.clamp(target, 0.0, 1.0)
    prediction = torch.clamp(prediction, 0.0, 1.0)
    mse = F.mse_loss(target, prediction)
    psnr_value = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr_value


Discriminator=Discriminator().to(device)
net_combined = nn.Sequential(CoarseSRNet(), DeepCNN()).to(device)


criterion = nn.MSELoss()
optimizer_G = optim.Adam(net_combined.parameters(), lr=0.0001, betas=(0.5, 0.999))
optimizer_Dis = optim.Adam(Discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))

dataset = SuperResolutionDataset('low', 'high')
data_loader = DataLoader(dataset, batch_size=2, shuffle=True)
num_epochs = 64

# generator_params = sum(p.numel() for p in net_combined.parameters())
# discriminator_params = sum(p.numel() for p in Discriminator.parameters())
# print("Generator parameters: ", generator_params)
# print("Discriminator parameters: ", discriminator_params)

checkpoint_dir = "new"
os.makedirs(checkpoint_dir, exist_ok=True)

def save_checkpoint(epoch, generator, discriminator, optimizer_G, optimizer_Dis):
    checkpoint = {
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_Dis_state_dict': optimizer_Dis.state_dict(),
    }
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

# Check if there are existing checkpoints
checkpoint_files = [file for file in os.listdir(checkpoint_dir) if file.endswith('.pt')]
if checkpoint_files:
    # Find the latest checkpoint file
    max_numeric_part = -1
    max_element = None
    for element in checkpoint_files:
        numeric_part = int(element.split('_')[2].split('.')[0])
        if numeric_part > max_numeric_part:
            max_numeric_part = numeric_part
            max_element = element
    checkpoint_path = os.path.join(checkpoint_dir, max_element)

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)
    start_epoch = checkpoint['epoch'] + 1
    net_combined.load_state_dict(checkpoint['generator_state_dict'])
    Discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
    optimizer_Dis.load_state_dict(checkpoint['optimizer_Dis_state_dict'])
    print(f"Resuming training from checkpoint: {checkpoint_path}")
else:
    start_epoch = 0

checkpoint_frequency = 1

result_file = "results.txt"

# Open the file in write mode
with open(result_file, "a") as file:
    file.write("Epoch\tD Loss\tG Loss\tPSNR\n") 
    file.close()

    for epoch in range(start_epoch,num_epochs):
        for i, data in enumerate(data_loader):
            file=open(result_file, "a")

            inputs, ground_truth = data
            batch_size = inputs.size(0)
            # Move data to GPU
            inputs = inputs.to(device)
            ground_truth = ground_truth.to(device)


            optimizer_Dis.zero_grad()
            real_labels = torch.ones(batch_size, 1).to(device)
            real_predictions = Discriminator(ground_truth)
            d_loss_real = criterion(real_predictions, real_labels)

            fake_images = net_combined(inputs)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            fake_predictions = Discriminator(fake_images.detach())
            d_loss_fake = criterion(fake_predictions, fake_labels)
                    
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_Dis.step()
            
            optimizer_G.zero_grad()
            targets = torch.ones(batch_size, 1).to(device)
            fake_predictions = Discriminator(fake_images)
            g_loss = criterion(fake_predictions, targets)
            g_loss.backward()
            optimizer_G.step()

            psnr_value = psnr(ground_truth, fake_images)
            d_loss_value = d_loss.item()
            g_loss_value = g_loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{i+1}/{len(data_loader)}] D Loss: {d_loss.item():.4f} G Loss: {g_loss.item():.4f} psnr: {psnr_value.item():.2f}")
        # Save the checkpoint
        if (epoch + 1) % checkpoint_frequency == 0:
            save_checkpoint(epoch, net_combined, Discriminator, optimizer_G, optimizer_Dis)
        # Save the values in the file
        file.write(f"{epoch+1}\t{d_loss_value:.4f}\t{g_loss_value:.4f}\t{psnr_value:.2f}\n")
        file.close()