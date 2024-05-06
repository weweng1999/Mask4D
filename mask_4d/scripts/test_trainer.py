from pytorch_lightning import Trainer

# Dummy values for testing
gpus = 1  # Assuming you have at least one GPU
max_epochs = 1
gradient_clip_val = 0.5

trainer = Trainer(
    gpus=gpus,
    max_epochs=max_epochs,
    gradient_clip_val=gradient_clip_val
)

# Print to confirm initialization
print("Trainer initialized successfully:", trainer)

