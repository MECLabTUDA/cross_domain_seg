# ------------------------------------------------------------------------------
# A standard segmentation agent, which performs softmax in the outputs.
# ------------------------------------------------------------------------------

from mp.agents.agent import Agent
from mp.eval.inference.predict import softmax
from mp.eval.accumulator import Accumulator
from mp.eval.evaluate import ds_losses_metrics
import torch
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import torchio
import random

class SDNetAgent(Agent):
    r"""An Agent for segmentation models."""
    def __init__(self, *args, **kwargs):
        if 'metrics' not in kwargs:
            kwargs['metrics'] = ['ScoreDice', 'ScoreIoU']
        super().__init__(*args, **kwargs)

    def get_inputs_targets(self, data, getName=False):
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
        r"""Applies a softmax transformation to the model outputs"""
        outputs = self.model(inputs)
        #outputs = softmax(outputs)
        return outputs

    def style_transfer(self, results, optimizer, optimizer_sdnet_z, loss_f, loss_MAE, train_dataloader, eval_datasets=dict(), load_domain_image_path=None, save_path=None):
        imgStyle = torchio.ScalarImage(load_domain_image_path)
        imgStyle = torchio.transforms.RescaleIntensity(out_min_max=(0, 1), percentiles=(0.1, 99.))(imgStyle)
        #Image(path=load_domain_image_path, type=torchio.INTENSITY)
        imgStyle = torch.Tensor.permute((imgStyle.data), (3, 0, 1, 2)).cuda()
        # torchio.transforms.RescaleIntensity(out_min_max=(0, 1), percentiles=(0.1, 99.))
        print(imgStyle.size())
        self.model.eval()

        # Run dest image
        self.model.forward(imgStyle)

        # Transfer remaining images

        print("Save in folder:")
        save_dest = os.path.join(save_path, 'TestImages', 'imagesTr')
        print(save_dest)
        for i, data in enumerate(train_dataloader):
            inputs, targets, name, x_affine = self.get_inputs_targets(data, getName = True)
            rec_x = self.model.forward_changeStyleToZ(inputs)
            rec_x = rec_x.permute((0, 2, 3, 1))
            img_rec = rec_x.data.detach().cpu().numpy() # .squeeze()
            self.save_synthetic_image(save_dest, str(name[0]), img_rec, x_affine)
        for i, data in enumerate(eval_datasets['test']):
            inputs, targets, name, x_affine = self.get_inputs_targets(data, getName = True)
            rec_x = self.model.forward_changeStyleToZ(inputs)
            rec_x = rec_x.permute((0, 2, 3, 1))
            img_rec = rec_x.data.detach().cpu().numpy() # .squeeze()
            self.save_synthetic_image(save_dest, str(name[0]), img_rec, x_affine)
        #for i, data in enumerate(eval_dataloader):
        #    self.model.forward(data)
        #    self.save_synthetic_image(save_dest)

        # Have images in Folder
        if False:
            for i, data in enumerate(train_dataloader):
                    inputs, targets, name, x_affine = self.get_inputs_targets(data, getName = True)
                    print(name)
                    rec_x = self.model.forward_changeStyleToZ(inputs)
                    #self.model.forward(inputs)

                    # Save Image rec
                    img_rec = self.model.fake_s.data[0][0].squeeze().detach().cpu().numpy()
                    img_rec = transforms.ToPILImage()(img_rec * 255).convert('RGB')
                    img = inputs.data[0][0].squeeze().detach().cpu().numpy()
                    img = transforms.ToPILImage()(img * 255).convert('RGB')
                    width, height = img_rec.size
                    result_img = Image.new(img_rec.mode, (width*2, height))
                    result_img.paste(img, (0, 0, width, height))
                    result_img.paste(img_rec, box=(width, 0))

                    name = str(random.randint(0, 1000)) + ".png"

                    result_img.save(os.path.join(save_path, 'TestImages', name))

    def save_synthetic_image(self, save_dest, name, img_rec, affine):
        os.makedirs(save_dest, exist_ok=True)
        img = torchio.Image(tensor=img_rec, affine=affine.squeeze())
        path = os.path.join(save_dest, name + ".nii.gz")
        #print(path)
        img.save(path)

    def train(self, results, optimizer, optimizer_sdnet_z, loss_f, loss_MAE, train_dataloader,
        init_epoch=0, nr_epochs=100, run_loss_print_interval=10,
        eval_datasets=dict(), eval_interval=10, 
        save_path=None, save_interval=10):
        r"""Train a model through its agent. Performs training epochs, 
        tracks metrics and saves model states.
        """
        if init_epoch == 0:
            self.model.eval()
            self.track_metrics(init_epoch, results, loss_f, loss_MAE, eval_datasets, save_path=save_path)
        for epoch in range(init_epoch, init_epoch+nr_epochs):
            self.model.train()
            print_run_loss = (epoch + 1) % run_loss_print_interval == 0
            print_run_loss = print_run_loss and self.verbose
            self.perform_training_epoch(optimizer, optimizer_sdnet_z, loss_f, loss_MAE, train_dataloader, 
                print_run_loss=print_run_loss)
        
            # Track statistics in results
            if (epoch + 1) % eval_interval == 0:
                self.model.eval()
                self.track_metrics(epoch + 1, results, loss_f, loss_MAE, eval_datasets, save_path=save_path)

            # Save agent and optimizer state
            if (epoch + 1) % save_interval == 0 and save_path is not None:
                self.save_state(save_path, 'epoch_{}'.format(epoch + 1), optimizer)

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
            fake_s, fake_z = self.model.forward_z(inputs)
            c_list.append(np.concatenate((fake_z.detach().cpu().flatten().numpy(),fake_s.detach().cpu().flatten().numpy()[0:30]), axis=0))
            #c_list.append(fake_z.detach().cpu().flatten().numpy())

        return z_list, c_list

    def getValuesForBetaVAEMetric(self, datalaoder, number_datasets):
        z_list = []
        c_list = []
        self.model.eval()

        firstImg = None
        firstZ = None
        #elements = 
        for i, data in enumerate(dataloader):
            inputs, _, z, _, _, __ = data
            if firstZ != z:
                continue
            if i % 2 == 0:
                firstImg = inputs
                firstZ = z
                continue
            
            # groundtruth
            z = self.one_hot(label=[z], depth=number_datasets)
            z_list.append(z.flatten().numpy())
            
            # 
            inputs = inputs.to(self.device)
            firstImg = firstImg.to(self.device)
            inputs = self.model.preprocess_input(inputs)  
            firstImg = self.model.preprocess_input(firstImg)  
            c_list.append(self.model.forward_z(inputs).detach().cpu().flatten().numpy())

        return z_list, c_list

    def getLosses(self, inputs, targets, loss_f, loss_MAE, save_pred=False, save_path=None):
        # Forward pass
        fake_m, rec_x, fake_z, divergence = self.get_outputs(inputs)

        # save training images
        if save_pred:
            
            # Save Mask
            mask_pred = fake_m.data[0][1].squeeze().detach().cpu().numpy()
            mask_pred = transforms.ToPILImage()(mask_pred * 255).convert('RGB')
            mask = targets.data[0][0].squeeze().detach().cpu().numpy()
            mask = transforms.ToPILImage()(mask * 255).convert('RGB')
            width, height = mask_pred.size
            result_img = Image.new(mask_pred.mode, (width*2, height))
            result_img.paste(mask, (0, 0, width, height))
            result_img.paste(mask_pred, box=(width, 0))
            result_img.save(save_path)

            # Save img rec
            img_rec = self.model.rec_x.data[0][0].squeeze().detach().cpu().numpy()
            img_rec = transforms.ToPILImage()(img_rec * 255).convert('RGB')
            img = inputs.data[0][0].squeeze().detach().cpu().numpy()
            img = transforms.ToPILImage()(img * 255).convert('RGB')
            width, height = img_rec.size
            result_img = Image.new(img_rec.mode, (width*2, height))
            result_img.paste(img, (0, 0, width, height))
            result_img.paste(img_rec, box=(width, 0))
            result_img.save(save_path)


        # Optimization step
        loss_mask = loss_f(fake_m, targets)
        loss_rec = loss_MAE(rec_x, inputs)
        divergence = torch.sum(divergence)
        
        loss_z = torch.Tensor(0)
        # z = z
        if True:
            _, fake_z_rec = self.model.forward_z(rec_x)
            loss_z = loss_MAE(fake_z_rec, fake_z)

        return loss_mask, loss_rec, divergence, loss_z

    def perform_training_epoch(self, optimizer, optimizer_sdnet_z, loss_f, loss_MAE, train_dataloader, 
        print_run_loss=False):
        r"""Perform a training epoch
        
        Args:
            print_run_loss (bool): whether a runing loss should be tracked and
                printed.
        """
        #print(train_dataloader)
        acc = Accumulator('loss')
        for _, data in enumerate(train_dataloader):
            # Get data
            inputs, targets = self.get_inputs_targets(data)
            loss_mask, loss_rec, divergence, loss_z = self.getLosses(inputs, targets, loss_f, loss_MAE)

            optimizer.zero_grad()
            loss = loss_mask + loss_rec + divergence + loss_z

            loss.backward()

            optimizer.step()

            #loss_mask, loss_rec, divergence, loss_z = self.getLosses(inputs, targets, loss_f, loss_MAE)
            #optimizer_sdnet_z.zero_grad()
            #loss = loss_z
            #loss.backward()

            #optimizer_sdnet_z.step()
            
            acc.add('loss', float(loss.detach().cpu()), count=len(inputs))

        if print_run_loss:
            print('\nRunning loss: {}'.format(acc.mean('loss')))

    def track_metrics(self, epoch, results, loss_f, loss_MAE, datasets, save_path=None):
        r"""Tracks metrics. Losses and scores are calculated for each 3D subject, 
        and averaged over the dataset.
        """
        for ds_name in datasets:
            count = 0
            scores = {
                "dice_mean": 0,
                "loss_rec_mean": 0,
                "divergence_mean": 0,
                "loss_z_mean": 0
            }

            length = datasets[ds_name].__len__()
            for i, data in enumerate(datasets[ds_name]):
                inputs, targets = self.get_inputs_targets(data)

                if i % int(length/5) == 0:
                    save_pred = True
                    save_path_loc = os.path.join(save_path, "epoch" + str(epoch) + "_image " + str(i) + ".png")
                else:
                    save_pred = False
                    save_path_loc = None
                loss_mask, loss_rec, divergence, loss_z = self.getLosses(inputs, targets, loss_f, loss_MAE, save_pred=save_pred, save_path=save_path_loc)

                scores["dice_mean"] += loss_mask.item()
                scores["loss_rec_mean"] += loss_rec.item()
                scores["divergence_mean"] += divergence.item()
                scores["loss_z_mean"] += loss_z.item()

                count += 1

            # Divide by count to get mean
            for s in scores:
                scores[s] /= count

            for s in scores:
                if scores[s] > 1:
                    scores[s] = 1
            
            # Add to results
            results.add(epoch=epoch, metric='Mean_Dice', data=ds_name, value=scores["dice_mean"])
            results.add(epoch=epoch, metric='Mean_Loss_Rec', data=ds_name, value=scores["loss_rec_mean"])
            results.add(epoch=epoch, metric='Mean_Divergence', data=ds_name, value=scores["divergence_mean"])
            results.add(epoch=epoch, metric='Mean_Loss_Z', data=ds_name, value=scores["loss_z_mean"])
                
            results.print_dataset_inEpoch(ds_name, epoch)

        if False:
            for ds_name, ds in datasets.items():
                eval_dict = ds_losses_metrics(ds, self, loss_f, loss_MAE, self.metrics)
                for metric_key in eval_dict.keys():
                    results.add(epoch=epoch, metric='Mean_'+metric_key, data=ds_name, 
                        value=eval_dict[metric_key]['mean'])
                    results.add(epoch=epoch, metric='Std_'+metric_key, data=ds_name, 
                        value=eval_dict[metric_key]['std'])
                if self.verbose:
                    print('Epoch {} dataset {}'.format(epoch, ds_name))
                    for metric_key in eval_dict.keys():
                        print('{}: {}'.format(metric_key, eval_dict[metric_key]['mean']))
