import torch


class Config:
    def __init__(self, 
                 model = "allenai/longformer-base-4096",
                 max_length =1024,
                 train_batch_size=1,
                 valid_batch_size=1,
                 epochs=5,
                 learning_rates=[0.25e-4, 0.25e-4, 0.25e-4, 0.25e-4, 0.25e-5],
                 max_grad_norm=10,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.max_length = max_length
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.epochs = epochs
        self.learning_rates = learning_rates
        self.max_grad_norm = max_grad_norm
        self.device = device