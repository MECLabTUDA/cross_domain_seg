from mp.agents.agent import Agent
import numpy as np
from mp.eval.inference.predict import softmax
import torch.nn as nn
import torchio

class UnetEvalAgent(Agent):
    r"""An Agent for autoencoder models."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_inputs_targets(self, data, getName=False):
        r"""Prepares a data batch.

        Args:
            data (tuple): a dataloader item, possibly in cpu

        Returns (tuple): preprocessed data in the selected device.
        """
        inputs, targets, _, _, name, x_affine = data 
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        inputs = self.model.preprocess_input(inputs)       

        if not getName:
            return inputs, targets.float()
        else:
            return inputs, targets.float(), name, x_affine

    def get_outputs(self, inputs):
        r"""Applies a softmax transformation to the model outputs"""
        outputs = self.model(inputs)
        outputs = softmax(outputs)
        return outputs

    def eval(self, results, optimizer, loss_f, train_dataloader, test_dataloader):
        self.model.eval()
        self.L1_criterion = nn.L1Loss()
        loss_background_all = []
        loss_foreground_all = []
        for i, data in enumerate(train_dataloader):            
            inputs, targets = self.get_inputs_targets(data)

            outputs = self.get_outputs(inputs)
            loss = loss_f(outputs, targets).item()
            loss_background_all.append(loss)

        print(np.array(loss_background_all).mean())

    def train(self, results, optimizer, loss_f, train_dataloader,
            init_epoch=0, nr_epochs=100, run_loss_print_interval=10,
            eval_dataloader=None, eval_interval=10, 
            save_path=None, save_interval=10):
            r"""Train a model through its agent. Performs training epochs, 
            tracks metrics and saves model states.
            """

            for epoch in range(init_epoch, init_epoch+nr_epochs):
                print("EPOCH: " + str(epoch))
                print_run_loss = (epoch + 1) % run_loss_print_interval == 0
                print_run_loss = print_run_loss and self.verbose

                nr_samples = 1
                samples = random.sample(range(len(train_dataloader)), nr_samples)
                loss_G = []
                for i, data in enumerate(train_dataloader):
                    if math.isnan(data[0][0][0][0][0]):
                        print("Skipped")
                        continue

                    inputs, targets = self.get_inputs_targets(data)
                    outputs = self.get_outputs(inputs)
                    
                    loss_G.append(i_loss_G)
                    loss_D.append(i_loss_D)

                    if i in samples:
                        self.model.save_result(save_path, epoch + 1, 'train')

                # Track statistics in results
                if (epoch + 1) % eval_interval == 0:
                    results.add(epoch + 1, 'Loss_G_Mean', stat.mean(loss_G), 'train')
                    results.add(epoch + 1, 'Loss_D_Mean', stat.mean(loss_D), 'train')
                    self.test(results, epoch, eval_dataloader, save_path)
                    #self.track_metrics(epoch + 1, results, loss_G, loss_D, eval_datasets)

                # Save agent and optimizer state
                if ((epoch + 1) % save_interval == 0 or init_epoch+nr_epochs == epoch) and save_path is not None:
                    print("SaveAgent")
                    self.save_state(save_path, 'epoch_{}'.format(epoch + 1), optimizer)