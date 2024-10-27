import torch
import torch.nn as nn
from models.GPT import GPTModel
from dataloader.dataloader import create_dataloader
from tqdm import tqdm
import os

class Trainer:
    def __init__(self, model: GPTModel, training_args):
        self.output_dir= training_args['output_dir']
        self.num_epochs = training_args['num_epochs']
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)

        # Load data from txt file
        self.data_path = training_args['data_path']
        with open(self.data_path, 'r', encoding= 'utf-8') as f:
            text_data = f.read()

        train_ratio = 0.9
        split_idx = int(train_ratio * len(text_data))

        self.train_loader = create_dataloader(txt= text_data[:split_idx],
                                              batch_size= training_args['train_batch_size'],
                                              max_length= 256,
                                              stride= 256,
                                              shuffle= True)
        
        self.val_loader = create_dataloader(txt= text_data[split_idx:],
                                            batch_size= training_args['val_batch_size'],
                                            max_length= 256,
                                            stride= 256,
                                            shuffle= False)
        
        self.optim = torch.optim.AdamW(self.model.parameters(), lr= training_args['learning_rate'], weight_decay= training_args['weight_decay'])

    def train(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        best_loss = float('inf')
        
        if os.path.exists(os.path.join(self.output_dir, "last_model.pth")):
            print("Loading lastest model to continue train!!!")
            checkpoint = torch.load(os.path.join(self.output_dir, "last_model.pth"))
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optim.load_state_dict(checkpoint['optim_state_dict'])
            best_loss = checkpoint['best_score']

        else:
            print("First time training!!!")
        
 
        for i in range(self.num_epochs):
            self.model.train() # Set model to train mode
            train_loss, val_loss = 0.
            for _, (input, target) in enumerate(tqdm(self.train_loader)):
                self.optim.zero_grad()
                input, target = input.to(self.device), target.to(self.device)
                _, loss = self.model(input, target)
                loss.backward()
                self.optim.step()
                train_loss += loss.item()

            train_loss /= len(self.train_loader)

            self.model.eval() # Set model to eval mode
            with torch.inference_mode():
                for _, (input, target) in enumerate(tqdm(self.val_loader)):
                    input, target = input.to(self.device), target.to(self.device)
                    _, loss = self.model(input, target)
                    val_loss += loss.item()

                val_loss /= len(self.val_loader)

                print(f"Epoch {i + 1}: "
                      f"Train loss: {train_loss:.4f}  |  Val loss: {val_loss:.4f}")
                
                # Save last model
                torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'optim_state_dict': self.optim.state_dict(),
                        'criterion': 'cross_entropy_loss',
                        'score': val_loss
                    }, os.path.join(self.output_dir, "last_model.pth"))
                
                # Save best model
                if val_loss <= best_loss:
                    print(f"Save best model with val loss {val_loss:.4f}")

                    best_loss = val_loss
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'optim_state_dict': self.optim.state_dict(),
                        'criterion': 'cross_entropy_loss',
                        'score': best_loss
                    }, os.path.join(self.output_dir, "best_model.pth"))