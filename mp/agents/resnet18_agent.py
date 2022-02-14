# ------------------------------------------------------------------------------
# An resnet18 agent.
# ------------------------------------------------------------------------------

from mp.agents.agent import Agent
import numpy as np
import torch
from mp.eval.inference.predict import softmax

class Resnet18Agent(Agent):
    r"""An Agent for resnet models."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_name = 'CrossEntropyLoss'
        self.output_labels = 0

    def get_outputs(self, inputs):
        r"""Returns model outputs.
        Args:
            data (torch.tensor): inputs

        Returns (torch.tensor): model outputs, with one channel dimension per 
            label.
        """
        output = self.model(inputs)
        output = softmax(output)
        return output

    def evaluate(self, train_dataloader=None, eval_dataloader=None, loss=None, output_labels=3):
        self.model.eval()

        self.evaluate_dataloader(train_dataloader, "Train set: ", loss, output_labels=output_labels)
        self.evaluate_dataloader(eval_dataloader, "Eval set: ", loss, output_labels=output_labels)

    def getMaxValuePos(self, array_np):
        max_value = -100000
        for i in range(np.shape(array_np)[1]):
            if max_value < array_np[0][i]:
                out_max = i
                max_value = array_np[0][i]

        return out_max


    def evaluate_dataloader(self, dataloader, text="", loss=None, output_labels=3):
        correctResults = 0
        dict = {}
        for i in range(output_labels):
            dict[i] = {}

        for i, data in enumerate(dataloader):
            inputs, targets = self.get_inputs_targets(data)

            outputs = self.get_outputs(inputs)

            outputs = outputs.cpu().detach().numpy()
            targets = targets.cpu().detach().numpy()

            max_value = -100000
            out_max = self.getMaxValuePos(outputs)

            if loss == 'L1Loss':
                tar_max = self.getMaxValuePos(targets)
            else:
                tar_max = targets[0]

            #out_max = np.where(outputs == np.amax(outputs)[1])
            
            #print(out_max)
            #print(tar_max)
            if out_max == tar_max:
                correctResults += 1
            if out_max in dict[tar_max]:
                dict[tar_max][out_max] += 1
            else:
                dict[tar_max][out_max] = 1
        
        print(text + str(correctResults / dataloader.__len__()))
        print(dict)

        return 

    def one_hot(self, label, depth):
        if not torch.is_tensor(label):
            label = torch.tensor([label])
        out_tensor = torch.zeros(len(label), depth, dtype=torch.float)
        for i, index in enumerate(label):
            out_tensor[i][index] = 1
        return out_tensor

    def get_inputs_targets(self, data, getName = False):
        r"""The usual dataloaders are used for autoencoders. However, these 
        ignore the target and instead treat he input as target
        """
        inputs, _, pose, _, _, _ = data
        inputs = inputs.to(self.device)
        inputs = self.model.preprocess_input(inputs)
        if self.loss_name == 'CrossEntropyLoss':
            if torch.is_tensor(pose):
                targets = pose
            else:
                targets = torch.tensor([pose]) #self.one_hot(pose, self.model.output_shape)
        if self.loss_name == 'L1Loss':
            #print(pose)
            targets = self.one_hot(pose, self.output_labels)
        inputs = inputs.cuda()
        targets = targets.cuda()
        #if getName == False:
        return inputs, targets
        #else:


    def predict_from_outputs(self, outputs):
        r"""No transformation is performed on the outputs, as the goal is to
        reconstruct the input. Therefore, the output should have the same dim
        as the input."""
        return outputs