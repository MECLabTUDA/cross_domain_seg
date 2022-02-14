# ------------------------------------------------------------------------------
# An Agent is an extension of a model. It also includes the logic to train the 
# model. This superclass diverges slightly from mp.agents.agent.Agent.
# ------------------------------------------------------------------------------

import os
import shutil

from numpy.lib.function_base import _average_dispatcher, average
from mp.eval.accumulator import Accumulator
from mp.utils.load_restore import pkl_dump, pkl_load
from mp.utils.pytorch.pytorch_load_restore import save_model_state, load_model_state, save_optimizer_state, load_optimizer_state
from mp.eval.inference.predict import arg_max
from mp.eval.evaluate import ds_losses_metrics
import torchio
import numpy as np
import torch
from mp.eval.metrics.scores import ScoreDice
from mp.eval.metrics.mean_scores import get_tp_tn_fn_fp_segmentation, get_mean_scores
import pickle
import surface_distance
import sys
#from surface_distance.surface_distance import metrics

class Agent:
    r"""An Agent, which includes a model and extended fields and logic.

    Args:
        model (mp.models.model.Model): a model
        label_names (list[str]): a list of label names
        metrics (list[str]): a list of metric names. Metric names are class 
            names for descendants of mp.eval.metrics.scores.ScoreAbstract.
            These are tracked by the track_metrics method.
        device (str): 'cpu' or a cuda-enabled gpu, e.g. 'cuda:0'
        scores_label_weights (tuple[float]): weights for each label to calculate
            metrics (not for the loss, which is defined in the loss definition).
            For instance, to explude "non-cares" from the metric calculation.
        verbose (bool): whether certain info. should be printed during training
    """
    def __init__(self, model, label_names=None, metrics=[], device='cuda:0', 
        scores_label_weights=None, verbose=True):
        self.model = model
        self.device = device
        self.metrics = metrics
        self.label_names = label_names
        self.nr_labels = len(label_names) if self.label_names else 0
        self.scores_label_weights = scores_label_weights
        self.verbose = verbose
        self.agent_state_dict = dict()

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

    def get_inputs_targets_full(self, data, getName=False):
        r"""Prepares a data batch.

        Args:
            data (tuple): a dataloader item, possibly in cpu

        Returns (tuple): preprocessed data in the selected device.
        """
        inputs, targets, _, _, name, x_affine = data #x, y, pose, identity, name, x_affine
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        inputs = self.model.preprocess_input(inputs)       

        if not getName:
            return inputs, targets.float()
        else:
            return inputs, targets.float(), name, x_affine

    def get_outputs(self, inputs):
        r"""Returns model outputs.
        Args:
            data (torch.tensor): inputs

        Returns (torch.tensor): model outputs, with one channel dimension per 
            label.
        """
        return self.model(inputs)

    def predict_from_outputs(self, outputs):
        r"""Returns argmaxed outputs.

        Args:
            data (torch.tensor): model outputs, with one channel dimension per 
            label.

        Returns (torch.tensor): a one-channeled prediction.
        """
        return arg_max(outputs, channel_dim=1)

    def predict(self, inputs):
        r"""Returns model outputs.
        Args:
            data (torch.tensor): inputs

        Returns (torch.tensor): a one-channeled prediction.
        """
        outputs = self.get_outputs(inputs)
        return self.predict_from_outputs(outputs)

    def perform_training_epoch(self, optimizer, loss_f, train_dataloader, 
        print_run_loss=False):
        r"""Perform a training epoch
        
        Args:
            print_run_loss (bool): whether a runing loss should be tracked and
                printed.
        """
        acc = Accumulator('loss')
        for i, data in enumerate(train_dataloader):
            # Get data
            inputs, targets = self.get_inputs_targets(data)

            # Forward pass
            outputs = self.get_outputs(inputs)
            
            # Optimization step
            optimizer.zero_grad()
            loss = loss_f(outputs, targets)
            loss.backward()
            optimizer.step()
            acc.add('loss', float(loss.detach().cpu()), count=len(inputs))

        if print_run_loss:
            print('\nRunning loss: {}'.format(acc.mean('loss')))

    def train(self, results, optimizer, loss_f, train_dataloader,
        init_epoch=0, nr_epochs=100, run_loss_print_interval=10,
        eval_datasets=dict(), eval_interval=10, 
        save_path=None, save_interval=10):
        r"""Train a model through its agent. Performs training epochs, 
        tracks metrics and saves model states.
        """
        if init_epoch == 0 or nr_epochs == 0:
            self.track_metrics(init_epoch, results, loss_f, eval_datasets)
        for epoch in range(init_epoch, init_epoch+nr_epochs):
            print_run_loss = True
            print_run_loss = print_run_loss and self.verbose
            self.perform_training_epoch(optimizer, loss_f, train_dataloader, 
                print_run_loss=print_run_loss)
        
            # Track statistics in results
            if (epoch + 1) % eval_interval == 0:
                self.track_metrics(epoch + 1, results, loss_f, eval_datasets)

            # Save agent and optimizer state
            if (epoch + 1) % save_interval == 0 and save_path is not None:
                self.save_state(save_path, 'epoch_{}'.format(epoch + 1), optimizer)

    def evaluate_by_patient(self, results, train_dataloader, test_dataloader, loss=None, test=False, coherence=False):
        self.model.eval()

        if test == False:
            self.evaluate_dataloader(train_dataloader, "train", loss, coherence)
            self.evaluate_dataloader(test_dataloader, "val", loss, coherence)
        else:
            self.evaluate_dataloader(train_dataloader, "test", loss, coherence)
        return 

    def evaluate_dataloader(self, dataloader, text="", loss=None, coherence=False):
        correctResults = 0

        patients = {}
        for i, data in enumerate(dataloader):
            inputs, targets, name, x_affine = self.get_inputs_targets_full(data, getName=True)

            outputs = self.get_outputs(inputs)


            inputs = inputs.cpu().detach().numpy()
            outputs = outputs.cpu().detach().numpy()

            # outputs
            outputs[outputs > 0.5] = 1
            outputs[outputs <= 0.5] = 0
            #print(outputs)

            targets = targets.cpu().detach().numpy()

            name_split = name[0].split('_')

            patient_name = name_split[0] + name_split[1]

            if patient_name not in patients:
                patients[patient_name] = {}

            patients[patient_name][int(name_split[2])] = {"inputs":inputs, "outputs":outputs, "targets":targets}

            patients[patient_name]["affine"] = x_affine[0]

        print(patients.keys())

        patients_combined = {}
        for p in patients.keys():
            patients_combined[p] = {}
            patients_combined[p]["inputs"] = self.combine_patient(patients[p], tag='inputs')
            patients_combined[p]["outputs"] = self.combine_patient(patients[p], tag='outputs')
            patients_combined[p]["targets"] = self.combine_patient(patients[p], tag='targets')

            for type in ["inputs", "outputs", "targets"]:
                name = p + "_" + type
                if type != "inputs":
                    tensor_save = patients_combined[p][type][1]
                    tensor_save = np.expand_dims(tensor_save, axis=0)
                else:
                    tensor_save = patients_combined[p][type]
                affineV = np.array([[1,0,0,0],[0,1,0,0],[0,0,5,0],[0,0,0,1]])
                torchio.Image(tensor=tensor_save, affine=affineV).save("M:/TestOut/" + name + ".nii.gz") #, affine=patients[p]["affine"]

        fullScore = 0
        patients_scores = {}
        for p in patients.keys():
            #output_loss = torch.from_numpy(patients_combined[p]["outputs"][1]).contiguous()
            #target_loss = torch.from_numpy(patients_combined[p]["targets"][1]).contiguous()
            
            score = get_mean_scores(torch.from_numpy(patients_combined[p]["outputs"][1]), torch.from_numpy(patients_combined[p]["targets"][1]),label_names=self.label_names)
            sys.setrecursionlimit(50000)
            #print(score)
            patients_scores[p] = score["ScoreDice[hippocampus]"]
            # Compute Surface distances
            if coherence:
                patients_combined[p]["outputs"][1], seperate_volumes = self.surface_count(patients_combined[p]["outputs"][1])#np.abs(1 - self.surface_count(patients_combined[p]["outputs"][1]) / self.surface_count(patients_combined[p]["targets"][1]))
                #surface_distances = surface_distance.compute_surface_distances(mask_gt=patients_combined[p]["targets"][1].astype(bool), mask_pred=patients_combined[p]["outputs"][1].astype(bool), spacing_mm=(1, 1, 1))
                a = seperate_volumes #np.size(surface_distances["distances_gt_to_pred"]) / np.size(surface_distances["distances_pred_to_gt"])
                #print(a) #distances_gt_to_pred
                patients_scores[p] = a

            print("------------")
            print(patients_scores[p])

            fullScore += patients_scores[p] #score["ScoreDice[hippocampus]"]
        file = "M:/TestOut/" + text + "Dice" + ".pkl"
        with open(file, 'wb') as f:
            pickle.dump(patients_scores, file=f)
        print("-----------------------------------------------------------------")
        print("Final Dice Score: ")
        fullScore = fullScore / len(patients)
        print(fullScore)

        # SAVE NEW MASKS
        for p in patients.keys():

            for type in ["outputs"]:
                name = p + "_" + type
                if type != "inputs":
                    tensor_save = patients_combined[p][type][1]
                    tensor_save = np.expand_dims(tensor_save, axis=0)
                else:
                    tensor_save = patients_combined[p][type]
                affineV = np.array([[1,0,0,0],[0,1,0,0],[0,0,5,0],[0,0,0,1]])
                torchio.Image(tensor=tensor_save, affine=affineV).save("M:/TestOut/" + name + ".nii.gz") #, affine=patients[p]["affine"]
        return 0

    def surface_count(self, mask):
        # Edge detection
        if False:
            for x in range(np.shape(mask)[0]-1):
                for y in range(np.shape(mask)[1]-1):
                    for z in range(np.shape(mask)[2]-1):
                        if mask[x][y][z] != 1:
                            continue
                        # check around
                        if x != 0 and x != np.shape(mask)[0]:
                            if mask[x+1][y][z] == 0:
                                mask[x][y][z] = 2
                                continue
                            if mask[x-1][y][z] == 0:
                                mask[x][y][z] = 2
                                continue
                        if y != 0 and y != np.shape(mask)[1]:
                            if mask[x][y+1][z] == 0:
                                mask[x][y][z] = 2
                                continue
                            if mask[x][y-1][z] == 0:
                                mask[x][y][z] = 2
                                continue
                        if z != 0 and z != np.shape(mask)[2]:
                            if mask[x][y][z+1] == 0:
                                mask[x][y][z] = 2
                                continue
                            if mask[x][y][z-1] == 0:
                                mask[x][y][z] = 2
                                continue

            mask[mask == 1] = 0
            mask[mask == 2] = 1

        #print(mask)
        count_volumes = 2
        for x in range(np.shape(mask)[0]-1):
            for y in range(np.shape(mask)[1]-1):
                for z in range(np.shape(mask)[2]-1):
                    if mask[x][y][z] == 1:
                        mask, volume = self.findCoherenceNoRecursion(x, y, z, mask, np.shape(mask), 0, 0, count_volumes)
                        if volume > 5:
                            count_volumes+=1
                        else:
                            mask[mask == count_volumes] = 0
                        #print(volume)
                        #mask[mask == 2] = 0
                    continue


                    if mask[x][y][z] != 1:
                        continue
                    # check around
                    if x != 0 and x != np.shape(mask)[0]:
                        if mask[x+1][y][z] == 0:
                            mask[x][y][z] = 2
                            continue
                        if mask[x-1][y][z] == 0:
                            mask[x][y][z] = 2
                            continue
                    if y != 0 and y != np.shape(mask)[1]:
                        if mask[x][y+1][z] == 0:
                            mask[x][y][z] = 2
                            continue
                        if mask[x][y-1][z] == 0:
                            mask[x][y][z] = 2
                            continue
                    if z != 0 and z != np.shape(mask)[2]:
                        if mask[x][y][z+1] == 0:
                            mask[x][y][z] = 2
                            continue
                        if mask[x][y][z-1] == 0:
                            mask[x][y][z] = 2
                            continue
        return mask, count_volumes-2 #np.count_nonzero(mask == 2)

    def findCoherenceNoRecursion(self, posx, posy, posz, mask, shape, depth, volume, volume_id):
        lastPos = []
        lastPos.append([posx, posy, posz])
        while(True):
            posx, posy, posz = lastPos[-1][0], lastPos[-1][1], lastPos[-1][2]

            if mask[posx][posy][posz] != volume_id:
                mask[posx][posy][posz] = volume_id
                volume += 1

            # X
            if posx > 0 and posx < np.shape(mask)[0]-1:
                if mask[posx+1][posy][posz] == 1:
                    lastPos.append([posx+1, posy, posz])
                    continue

            # Y
            if posy > 0 and posy < np.shape(mask)[1]-1:
                if mask[posx][posy+1][posz] == 1:
                    lastPos.append([posx, posy+1, posz])
                    continue

            # Z
            if posz > 0 and posz < np.shape(mask)[2]-1:
                if mask[posx][posy][posz+1] == 1:
                    lastPos.append([posx, posy, posz+1])
                    continue

            # Z
            if posz > 0 and posz < np.shape(mask)[2]-1:
                if mask[posx][posy][posz-1] == 1:
                    lastPos.append([posx, posy, posz-1])
                    continue

            # Y
            if posy > 0 and posy < np.shape(mask)[1]-1:
                if mask[posx][posy-1][posz] == 1:
                    lastPos.append([posx, posy-1, posz])
                    continue
    
            # X
            if posx > 0 and posx < np.shape(mask)[0]-1:
                if mask[posx-1][posy][posz] == 1:
                    lastPos.append([posx-1, posy, posz])
                    continue

            lastPos.pop()

            if lastPos.__len__() == 0:
                return mask, volume



        return mask, volume

    def findCoherence(self, posx, posy, posz, mask, shape, depth, volume, volume_id):
        mask[posx][posy][posz] = volume_id

        if depth >= 3000:
            return mask, volume
        # X
        if posx != 0 and posx != np.shape(mask)[0]:
            if mask[posx+1][posy][posz] == 1:
                mask, volume = self.findCoherence(posx+1, posy, posz, mask, shape, depth+1, volume+1, volume_id)

        # Y
        if posy != 0 and posy != np.shape(mask)[1]:
            if mask[posx][posy+1][posz] == 1:
                mask, volume = self.findCoherence(posx, posy+1, posz, mask, shape, depth+1, volume+1, volume_id)

        # Z
        if posz != 0 and posz != np.shape(mask)[2]:
            if mask[posx][posy][posz+1] == 1:
                mask, volume = self.findCoherence(posx, posy, posz+1, mask, shape, depth+1, volume+1, volume_id)

        # Z
        if posz != 0 and posz != np.shape(mask)[2]:
            if mask[posx][posy][posz-1] == 1:
                mask, volume = self.findCoherence(posx, posy, posz-1, mask, shape, depth+1, volume+1, volume_id)

        # Y
        if posy != 0 and posy != np.shape(mask)[1]:
            if mask[posx][posy-1][posz] == 1:
                mask, volume = self.findCoherence(posx, posy-1, posz, mask, shape, depth+1, volume+1, volume_id)
  
        # X
        if posx != 0 and posx != np.shape(mask)[0]:
            if mask[posx-1][posy][posz] == 1:
                mask, volume = self.findCoherence(posx-1, posy, posz, mask, shape, depth+1, volume+1, volume_id)



        return mask, volume

    def Dice(self, tp, tn, fn, fp):
        if tp == 0:
            if fn+fp > 0:
                return 0.
            else:
                return 1.
        return (2*tp)/(2*tp+fp+fn)

    def combine_patient(self, dictionary, tag=None):

        combined_patient = []
        for i in range(len(dictionary)-1):
            if i == 0:
                combined_patient = dictionary[i][tag]
            else:
                combined_patient = np.append(combined_patient, np.atleast_3d(dictionary[i][tag]), axis=0)#np.dstack((combined_patient, dictionary[i][tag]))

        combined_patient = np.swapaxes(combined_patient,0,1)
        combined_patient = np.swapaxes(combined_patient,1,2)
        combined_patient = np.swapaxes(combined_patient,2,3)

        #print(np.shape(combined_patient))

        return combined_patient






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

    def save_state(self, states_path, state_name, optimizer=None, overwrite=False):
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
            if optimizer is not None:
                save_optimizer_state(optimizer, 'optimizer', state_full_path)


    def restore_state(self, states_path, state_name, optimizer=None):
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
            if optimizer is not None: 
                load_optimizer_state(optimizer, 'optimizer', state_full_path, device=self.device)
            if self.verbose:
                print('State {} was restored'.format(state_name))
            return True
        except:
            print('State {} could not be restored'.format(state_name))
            return False