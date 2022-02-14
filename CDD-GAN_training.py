
# ------------------------------------------------------------------------------
# Trains a CDD-Gan network
# ------------------------------------------------------------------------------

# 1. Imports
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from mp.experiments.experiment import Experiment
from mp.data.data import Data
from mp.data.datasets.ds_hippocampus import Hippocampus
from mp.data.pytorch.pytorch_cddGan_dataset import PytorchCDDGanDataset
from mp.models.segmentation.unet_fepegar import UNet2D
from mp.models.CDD_GAN import CDD_GAN
from mp.eval.losses.losses_segmentation import LossClassWeighted, LossDiceBCE
from mp.agents.cddgan_agent import CDDGan_Agent
from mp.agents.segmentation_agent import SegmentationAgent
from mp.eval.result import Result
from datetime import datetime
from mp.paths import model_paths
import time 

# 2. Define configuration
config = {
    # Basic parameters
    'experiment_name':'CDD-GAN_TestT', 
    'device':'cuda:0', 
    'init_epoch': 0, 
    'nr_epochs': 3000,
    'nr_runs': 1, 
    'val_ratio': 0.0, 
    'test_ratio': 0.3,
    'input_shape': (1, 64, 64), 
    'resize': False, 
    'augmentation': 'none',
    'dataset':'Hippocampus_all',
    'domain_prefixes':['b', 'e', 's'], # the domain is identified by the prefix of the file loaded
    # Training parameters
    'class_weights': (0.,1.), 
    'lr_G': 0.0001, 'lr_D':0.0001, 
    'batch_size': 16,
    'beta1': 0.5, 'beta2': 0.999, 'w_L1': 1,
    'UseCycleLoss': True,
    'CycleLossFactor': 1, # 1, 3, 10
    # Dataset parameters
    'number_domain': 3, 
    'number_identity': 322, 
    'number_noise': 50, 
    # Load Unet for CDD-GAN_M
    'UseMaskDiscriminator': False,
    'unet_load': 'Unet_trained_standard',
    'unet_state_name': "epoch_200",
    'Unet_training_domain': 's',
    'MaskCELoss': True, # BCE or L1
    # Instead of training: Save the synthetically created images
    'save_synthetic_images': False,
    }

if config['save_synthetic_images'] == True:
    config['batch_size'] = 1
    config['unet_load'] = 'Unet_trained_standard_hipp_new'
    config['dataset'] = 'Hippocampus_warp_test'

# Unet training domain number
config['Unet_training_domain'] = config['domain_prefixes'].index(config['Unet_training_domain'])

#config['batch_size'] = config['batch_size'] * config['number_pose']
device = config['device']
if config['device'] != 'cpu':
    device_name = torch.cuda.get_device_name(device)
    print('Device name: {}'.format(device_name))
else:
    print('Device name: CPU')
input_shape = config['input_shape']  
reload_exp = True if (config['init_epoch'] > 0) else False


# 3. Create experiment directories
exp = Experiment(config=config, name=config['experiment_name'], notes='', reload_exp=reload_exp)

# 4. Define data
data = Data()
data.add_dataset(Hippocampus(merge_labels=True, global_name=config['dataset'], domain_prefixes=config['domain_prefixes'])) #    'domain_prefixes':['b', 'e', 's'],
nr_labels = data.nr_labels
label_names = data.label_names
train_ds = (config['dataset'], 'train')
test_ds = (config['dataset'], 'test')

# 5. Create data splits for each repetition
if config['save_synthetic_images']:
    exp.new_data_splits(data)
else:
    exp.set_data_splits(data) #if reload_exp is False : 

# Now repeat for each repetition
for run_ix in range(config['nr_runs']):
    exp_run = exp.get_run(run_ix=run_ix, reload_exp_run=reload_exp)

    # 6. Bring data to Pytorch format
    datasets = dict()
    for ds_name, ds in data.datasets.items():
        for split, data_ixs in exp.splits[ds_name][exp_run.run_ix].items():
            if len(data_ixs) > 0:  # Sometimes val indexes may be an empty list
                aug = config['augmentation'] if not('test' in split) else 'none'
                datasets[(ds_name, split)] = PytorchCDDGanDataset(ds, 
                    ix_lst=data_ixs, size=input_shape, aug_key=aug, 
                    resize=config['resize'])


    # 7. Build train dataloader, and visualize
    dl = DataLoader(datasets[(train_ds)], 
        batch_size=config['batch_size'], shuffle=True)

    dl_test = DataLoader(datasets[(test_ds)], 
        batch_size=config['batch_size'], shuffle=True)

    # Load seg model
    if config['UseMaskDiscriminator']:
        model_seg = UNet2D(input_shape, nr_labels, num_encoding_blocks=5)#, out_channels_first_layer=64)
        model_seg.to(device)
        agent_seg = SegmentationAgent(model=model_seg, label_names=label_names, device=device)
        agent_seg.restore_state(states_path=model_paths[config['unet_load']], state_name=config['unet_state_name'])

        model_seg = agent_seg.model
    else:
        model_seg = None

    # 8. Initialize model
    model = CDD_GAN(number_domain=config['number_domain'], number_identity=config['number_identity'], number_noise=config['number_noise'], batch_size=config['batch_size'], config=config, segmentor=model_seg)
    model.init_weights()
    model.to(device)

    # 9. Define loss and optimizer
    loss_g = LossDiceBCE(bce_weight=1., smooth=1., device=device)
    loss_f = LossClassWeighted(loss=loss_g, weights=config['class_weights'], 
        device=device)
    optimizer_G = model.optimizer_G
    optimizer_D = model.optimizer_D

    # 10. Train model
    results = Result(name='training_trajectory')   
    agent = CDDGan_Agent(model=model, label_names=label_names, device=device)
    # Restore state
    #if config['init_epoch'] > 0:
    #    agent.restore_state(exp_run.paths['states'], 'epoch_' + str(config['init_epoch']), optimizer_G, optimizer_D)

    if config['init_epoch'] > 0:
        #results = exp_run.load_results()
        results = Result(name='training_trajectory')
        agent.restore_state(states_path=exp_run.paths['states'], state_name='epoch_' + str(config['init_epoch']), optimizer_G=optimizer_G, optimizer_D=optimizer_D)
    else: 
        results = Result(name='training_trajectory') 

    if not config['save_synthetic_images']:
        agent.train(results, optimizer_G, optimizer_D, loss_f, train_dataloader=dl,
            init_epoch=config['init_epoch'], nr_epochs=config['nr_epochs'], run_loss_print_interval=5,
            eval_dataloader=dl_test, eval_interval=25,
            save_path=exp_run.paths['states'], save_interval=25)

        # 11. Save and print results for this experiment run
        exp_run.finish(results=results, plot_metrics=['Loss_G_Mean', 'Loss_D_Mean'])#'Mean_ScoreDice', 'Mean_ScoreDice[prostate]'])
        test_ds_key = 'test'#'_'.join(test_ds)
        metric = 'Loss_G_Mean'
        results.get_max_epoch(metric, data=test_ds_key)
        last_dice = results.get_epoch_metric(
            results.get_max_epoch(metric, data=test_ds_key), metric, data=test_ds_key)
        print('Last Dice score for prostate class: {}'.format(last_dice))
    else:
        agent.save_synthetic_images(dl, dl_test, 'Hippocampus')
        print('Synthetic images saved.')

print(datetime.now())

