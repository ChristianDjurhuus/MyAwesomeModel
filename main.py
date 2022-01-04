import argparse
import sys

import torch
import numpy as np
from data import mnist
from model import MyAwesomeModel
import matplotlib.pyplot as plt


class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>"
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()
    
    def train(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--lr', default=0.1)
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement training loop here
        model = MyAwesomeModel()
        train_set, _ = mnist()
        optimizer = torch.optim.Adam(model.parameters())
        criterion = torch.nn.NLLLoss()
        epochs = 50
        running_loss = 0
        model.train()
        train_loss = []
        for e in range(epochs):
            for images, labels in train_set:
                #images = images.view(images.shape[0], -1)
                images = images.unsqueeze(1)
                optimizer.zero_grad()
                ps = model(images)
                loss = criterion(ps, labels)
                loss.backward()
                optimizer.step()

                train_loss.append(loss.item())

                # print statistics
                running_loss += loss.item()

            print('[%d] loss: %.3f' %
                  (e + 1, running_loss / len(train_set)))
            running_loss = 0.0



        plt.plot(train_loss, label = "Training loss")
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig("loss.png")

        torch.save(model.state_dict(), 'trained_model.pt')

    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('load_model_from', default="")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement evaluation logic here
        model = MyAwesomeModel()
        state_dict = torch.load(args.load_model_from)
        model.load_state_dict(state_dict)
        _, test_set = mnist()
        model.eval()
        accuracies = []
        with torch.no_grad():
            for images, labels in test_set:
                images = images.unsqueeze(1)
                #images = images.view(images.shape[0], -1)
                ps = model(images)
                #ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy = torch.mean(equals.type(torch.FloatTensor))
                #print(f'Accuracy: {accuracy.item() * 100}%')
                accuracies.append(accuracy)
        print('Estimate of accuracy: ', np.mean(accuracies))

if __name__ == '__main__':
    TrainOREvaluate()
    
    
    
    
    
    
    
    
    