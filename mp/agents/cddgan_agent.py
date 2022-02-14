# ------------------------------------------------------------------------------
# CDD agent, which performs softmax in the outputs.
# ------------------------------------------------------------------------------

from mp.agents.agent import Agent
from mp.eval.inference.predict import softmax
from mp.utils.pytorch.pytorch_load_restore import save_model_state, load_model_state, save_optimizer_state, load_optimizer_state
from mp.utils.load_restore import pkl_dump, pkl_load
import numpy as np
from tqdm import tqdm
import statistics as stat
import os
import math
import random
import torchio
import mp.paths
import torch

class CDDGan_Agent(Agent):
    r"""An Agent for segmentation models."""
    def __init__(self, *args, **kwargs):
        if 'metrics' not in kwargs:
            kwargs['metrics'] = ['ScoreDice', 'ScoreIoU']
        super().__init__(*args, **kwargs)

    def get_outputs(self, inputs):
        r"""Applies a softmax transformation to the model outputs"""
        outputs = self.model(inputs)
        #outputs = softmax(outputs)
        return outputs

    def get_inputs_targets(self, data):
        r"""Prepares a data batch.

        Args:
            data (tuple): a dataloader item, possibly in cpu

        Returns (tuple): preprocessed data in the selected device.
        """
        inputs, targets = data
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        inputs = self.model.preprocess_input(inputs)       
        return inputs, targets.float()

    def track_metrics(self, epoch, results, loss_f, datasets):
        r"""Tracks metrics. Losses and scores are calculated for each 3D subject, 
        and averaged over the dataset.
        """
        for ds_name, ds in datasets.items():
            eval_dict = ds_losses_metrics(ds, self, loss_f, self.metrics)
            for metric_key in eval_dict.keys():
                results.add(epoch=epoch, metric='Mean_'+metric_key, data=ds_name, 
                    value=eval_dict[metric_key]['mean'])
                results.add(epoch=epoch, metric='Std_'+metric_key, data=ds_name, 
                    value=eval_dict[metric_key]['std'])
            if self.verbose:
                print('Epoch {} dataset {}'.format(epoch, ds_name))
                for metric_key in eval_dict.keys():
                    print('{}: {}'.format(metric_key, eval_dict[metric_key]['mean']))

    # Allows for saving a list of optimizers if necessary
    def save_state(self, states_path, state_name, optimizers=None, optimizer_names=None, overwrite=False):
        r"""Saves an agent state. Raises an error if the directory exists and 
        overwrite=False.
        """
        if states_path is not None:
            state_full_path = os.path.join(states_path, state_name)
            if os.path.exists(state_full_path):
                if not overwrite:
                    raise FileExistsError
                shutil.rmtree(state_full_path)
            os.makedirs(state_full_path)
            save_model_state(self.model, 'model', state_full_path)
            pkl_dump(self.agent_state_dict, 'agent_state_dict', state_full_path)
            for idx, optimizer in enumerate(optimizers):
                save_optimizer_state(optimizer, 'optimizer_' + optimizer_names[idx], state_full_path)

    def restore_state(self, states_path, state_name, optimizer_G=None, optimizer_D=None):
        r"""Tries to restore a previous agent state, consisting of a model 
        state and the content of agent_state_dict. Returns whether the restore 
        operation  was successful.
        """
        state_full_path = os.path.join(states_path, state_name)
        try:
            correct_load = load_model_state(self.model, 'model', state_full_path, device=self.device)
            assert correct_load
            agent_state_dict = pkl_load('agent_state_dict', state_full_path)
            assert agent_state_dict is not None
            self.agent_state_dict = agent_state_dict
            if optimizer_G is not None: 
                load_optimizer_state(optimizer_G, 'optimizer_G', state_full_path, device=self.device)
            if optimizer_D is not None: 
                load_optimizer_state(optimizer_D, 'optimizer_D', state_full_path, device=self.device)
            if self.verbose:
                print('State {} was restored'.format(state_name))
            return True
        except:
            print('State {} could not be restored'.format(state_name))
            return False

    def save_synthetic_images(self, train_dataloader, eval_dataloader, dataset):
        print("Save in folder:")
        save_dest = mp.paths.original_data_paths[dataset] + "_syn/imagesTr"
        print(save_dest)
        for i, data in enumerate(eval_dataloader):
            self.model.forward(data, randomOut = False, outPose = 2)
            self.save_synthetic_image(save_dest)
        for i, data in enumerate(train_dataloader):
            self.model.forward(data, randomOut = False, outPose = 2)
            self.save_synthetic_image(save_dest)

    def save_synthetic_image(self, save_dest):
        os.makedirs(save_dest, exist_ok=True)
        for i, _ in enumerate(self.model.syn_image):
            #print(self.model.syn_image[i].data.unsqueeze(3).cpu().size())
            img = torchio.Image(tensor=self.model.syn_image[i].data.unsqueeze(3).cpu(), affine=self.model.affine[i].data.cpu().numpy())
            path = os.path.join(save_dest, self.model.name[i] + ".nii.gz")
            img.save(path)

    def test(self, results, epoch, eval_dataloader, save_path, nr_samples=1):
        samples = random.sample(range(len(eval_dataloader)), nr_samples)
        for i, data in enumerate(eval_dataloader):
            loss_G = []
            loss_D = []
            self.model.forward(data)
            i_loss_G, i_loss_D = self.model.optimize_DG_parameters(update_Weights=False)    
            loss_G.append(i_loss_G)
            loss_D.append(i_loss_D)
            if i in samples:
                self.model.save_result(save_path, epoch + 1, 'test')

            results.add(epoch + 1, 'Loss_G_Mean', stat.mean(loss_G), 'test')
            results.add(epoch + 1, 'Loss_D_Mean', stat.mean(loss_D), 'test')

    def one_hot(self, label, depth):
        r"""Return one hot encoding of label with depth."""
        out_tensor = torch.zeros(len(label), depth)
        for i, index in enumerate(label):
            out_tensor[i][index] = 1
        return out_tensor

    def getValuesForRegression(self, dataloader, number_datasets):
            z_list = []
            c_list = []
            self.model.eval()
            for _, data in enumerate(dataloader):
                inputs, _, z, _, _, __ = data
                z = self.one_hot(label=[z], depth=number_datasets)
                z_list.append(z.flatten().numpy())
                inputs = inputs.to(self.device)
                inputs = self.model.preprocess_input(inputs)  
                enc, real_z = self.model.forward_z(data)
                real_z = real_z.detach().cpu().flatten().numpy()
                real_z_id = np.copy(real_z)
                real_z_id[np.where(real_z!=np.max(real_z))] = 0
                real_z_id[np.where(real_z==np.max(real_z))] = 1
                enc = enc.detach().cpu().flatten().numpy()
                c_list.append(np.concatenate((real_z_id, enc), axis=0))

            return z_list, c_list

    def train(self, results, optimizer_G, optimizer_D, loss_f, train_dataloader,
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
            loss_D = []
            for i, data in enumerate(train_dataloader):
                #print(data)
                if math.isnan(data[0][0][0][0][0]):
                    print("Skipped")
                    continue
                self.model.forward(data)
                if(i % 20 == 0): # interval todo
                    i_loss_G, i_loss_D = self.model.optimize_DG_parameters()
                else:
                    i_loss_G, i_loss_D = self.model.optimize_G_parameters()
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
                self.save_state(save_path, 'epoch_{}'.format(epoch + 1), [optimizer_G, optimizer_D], ['G', 'D'])