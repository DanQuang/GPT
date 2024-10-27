from models.GPT import GPTModel
from tasks.train import Trainer

def main():
    # define training arguments
    training_args = {
    'output_dir': './my_model',
    'data_path': './data/the-verdict.txt',
    'num_epochs': 5,
    'train_batch_size': 16,
    'val_batch_size': 16,
    'learning_rate': 1e-5,
    'weight_decay': 0.01 # Use with AdamW
    }
    # init model
    model = GPTModel()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")

    trainer = Trainer(model= model, 
                      training_args= training_args)
    
    trainer.train()

if __name__ == "__main__":
    main()