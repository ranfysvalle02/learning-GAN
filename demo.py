import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np

# Hyperparameters
BATCH_SIZE = 64
LATENT_DIM = 100
EPOCHS = 200
LEARNING_RATE = 0.0002
IMG_SIZE = 28

# Data Preparation (Generating simple shapes: colored triangles)
def generate_colored_triangles_dataset(num_samples=1000, img_size=IMG_SIZE):
    data = []
    labels = []
    for _ in range(num_samples):
        image = np.zeros((3, img_size, img_size))  # RGB channels
        color = np.random.choice(['red', 'green', 'blue'])
        rr, cc = np.ogrid[:img_size, :img_size]
        for i in range(img_size // 4, 3 * img_size // 4):
            image[:, i, img_size // 2 - (i - img_size // 4):img_size // 2 + (i - img_size // 4) + 1] = 1.0

        if color == 'red':
            image[1:, :, :] = 0  # Only red channel is active
        elif color == 'green':
            image[0, :, :] = 0  # Only green channel is active
            image[2, :, :] = 0
        elif color == 'blue':
            image[:2, :, :] = 0  # Only blue channel is active

        # Normalize the image to be between -1 and 1
        image = (image * 2) - 1

        data.append(image.flatten())
        labels.append(0)  # Since we're only generating triangles, all labels are 0

    data = np.array(data, dtype=np.float32)
    return torch.tensor(data)

data = generate_colored_triangles_dataset()
dataset = TensorDataset(data)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Generator Model
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, IMG_SIZE * IMG_SIZE * 3),  # Output for RGB image
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z).view(-1, 3, IMG_SIZE, IMG_SIZE)

# Discriminator Model
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(IMG_SIZE * IMG_SIZE * 3, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)

# GAN Lightning Module
class GAN(pl.LightningModule):
    def __init__(self, latent_dim):
        super().__init__()
        self.generator = Generator(latent_dim)
        self.discriminator = Discriminator()
        self.latent_dim = latent_dim
        self.automatic_optimization = False

    def forward(self, z):
        return self.generator(z)

    def training_step(self, batch, batch_idx):
        real_data = batch[0].to(self.device)
        batch_size = real_data.size(0)

        # Sample noise and generate fake data
        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        fake_data = self(z)

        # Get optimizers
        d_optimizer, g_optimizer = self.optimizers()

        # Train discriminator
        d_optimizer.zero_grad()
        real_preds = self.discriminator(real_data)
        fake_preds = self.discriminator(fake_data.detach())
        real_loss = nn.functional.binary_cross_entropy(real_preds, torch.ones_like(real_preds))
        fake_loss = nn.functional.binary_cross_entropy(fake_preds, torch.zeros_like(fake_preds))
        d_loss = (real_loss + fake_loss) / 2
        self.manual_backward(d_loss)
        d_optimizer.step()
        self.log("d_loss", d_loss, prog_bar=True)

        # Train generator
        g_optimizer.zero_grad()
        fake_preds = self.discriminator(fake_data)
        g_loss = nn.functional.binary_cross_entropy(fake_preds, torch.ones_like(fake_preds))
        self.manual_backward(g_loss)
        g_optimizer.step()
        self.log("g_loss", g_loss, prog_bar=True)

    def configure_optimizers(self):
        lr = LEARNING_RATE
        d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
        g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=lr)
        return [d_optimizer, g_optimizer]

# Training the GAN
model = GAN(latent_dim=LATENT_DIM)
trainer = pl.Trainer(max_epochs=EPOCHS)
trainer.fit(model, dataloader)

# Generate and visualize an image
def generate_image(model, latent_dim, num_images=5):
    model.eval()
    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
    for i in range(num_images):
        z = torch.randn(1, latent_dim)
        with torch.no_grad():
            generated_image = model.generator(z).squeeze().cpu().numpy()
            generated_image = np.transpose(generated_image, (1, 2, 0))  # Convert to HxWxC
        axes[i].imshow((generated_image + 1) / 2)  # Rescale to [0, 1] for visualization
        axes[i].axis('off')
    plt.suptitle("Generated Shapes: Colored Triangles", fontsize=16)
    plt.show()

# Generate multiple images using the trained generator
generate_image(model, LATENT_DIM)

# Explain the generated images
def explain_gan_results():
    print("\nGANs (Generative Adversarial Networks) are powerful tools that learn to generate new data based on training data. In this example, the GAN is trained to generate colored triangles.")
    print("The generator learns to create images that resemble the training triangles with different colors, while the discriminator learns to distinguish between real shapes and those generated by the generator.")
    print("As training progresses, the generator becomes better at creating realistic colored triangles, while the discriminator becomes more adept at detecting fakes. This competition drives the improvement of both models.")
    print("\nNotice how the generated triangles may still have imperfections, but they often resemble the target shapes quite well. This demonstrates the ability of GANs to learn and replicate complex patterns from relatively simple data.")
    print("\nBy introducing color variations, we make the task more challenging and showcase the potential of GANs to learn more diverse and creative structures.")

# Explain the results to the user
explain_gan_results()
