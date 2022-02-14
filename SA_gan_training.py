# 1. Imports
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from mp.experiments.experiment import Experiment
from mp.data.data import Data
from mp.data.datasets.ds_hippocampus import Hippocampus
from mp.data.pytorch.pytorch_seg_dataset import PytorchSeg2DDataset
from mp.data.pytorch.pytorch_cddGan_dataset import PytorchCDDGanDataset
from mp.models.segmentation.unet_fepegar import UNet2D
from mp.models.sa_gan_model import SA_Full
from mp.eval.losses.losses_segmentation import LossClassWeighted, LossDiceBCE
from mp.agents.sa_agent import SA_Agent
from mp.agents.segmentation_agent import SegmentationAgent
from mp.eval.result import Result
from mp.paths import model_paths

# 2. Define configuration
config = {'experiment_name':'SA_hippocampus_1', 'device':'cuda:0', 'init_epoch': 0,
    'nr_runs': 1, 'cross_validation': False, 'val_ratio': 0.0, 'test_ratio': 0.3,
    'input_shape': (1, 64, 64), 'resize': False, 'augmentation': 'none', 
    'lr': 0.0001,
    'beta1':0.5,
    'beta2':0.999,
    'class_weights': (0.,1.), 'batch_size': 96, 'unet_state_name': "epoch_200",
    'dataset_s':'Hippocampus_standard',
    'dataset_t':'Hippocampus_hue',
    'style_transfer': False,
    'new_datasplit':False,
    'unet_model_path':'Unet_trained_standard',
    'domain_prefixes':['b', 'e', 's'],
    }

if config['style_transfer'] == True:
    config["new_datasplit"] = True
    config["batch_size"] = 1

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

# 4. Define - data_s
data_s = Data()
data_s.add_dataset(Hippocampus(merge_labels=True, global_name=config['dataset_s'], domain_prefixes=config['domain_prefixes']))
nr_labels = data_s.nr_labels
label_names = data_s.label_names
train_ds_s = (config['dataset_s'], 'train')
test_ds_s = (config['dataset_s'], 'test')

# 4. Define - data_t
data_t = Data()
data_t.add_dataset(Hippocampus(merge_labels=True, global_name=config['dataset_t'], domain_prefixes=config['domain_prefixes']))
nr_labels = data_t.nr_labels
label_names = data_t.label_names
train_ds_t = (config['dataset_t'], 'train')
test_ds_t = (config['dataset_t'], 'test')

# 5. Create data splits for each repetition
if config['new_datasplit'] is False:
    exp.set_data_splits(data_s)
    exp.set_data_splits_second(data_t)
else:
    exp.new_data_splits(data_s)
    exp.new_data_splits_second(data_t)

# Now repeat for each repetition
for run_ix in range(config['nr_runs']):
    exp_run = exp.get_run(run_ix=run_ix, reload_exp_run=reload_exp)

    # 6. Bring data to Pytorch format - data_s
    datasets_s = dict()
    for ds_name, ds in data_s.datasets.items():
        for split, data_ixs in exp.splits[ds_name][exp_run.run_ix].items():
            if len(data_ixs) > 0:  # Sometimes val indexes may be an empty list
                aug = config['augmentation'] if not('test' in split) else 'none'
                datasets_s[(ds_name, split)] = PytorchCDDGanDataset(ds, 
                    ix_lst=data_ixs, size=input_shape, aug_key=aug, 
                    resize=config['resize'])

    # 6. Bring data to Pytorch format - data_t
    datasets_t = dict()
    for ds_name, ds in data_t.datasets.items():
        for split, data_ixs in exp.splits_second[ds_name][exp_run.run_ix].items():
            if len(data_ixs) > 0:  # Sometimes val indexes may be an empty list
                aug = config['augmentation'] if not('test' in split) else 'none'
                datasets_t[(ds_name, split)] = PytorchCDDGanDataset(ds, 
                    ix_lst=data_ixs, size=input_shape, aug_key=aug, 
                    resize=config['resize'])


    # 7. Build train dataloader, and visualize - data_s
    dl_s = DataLoader(datasets_s[(train_ds_s)], 
        batch_size=config['batch_size'], shuffle=True)

    dl_test_s = DataLoader(datasets_s[(test_ds_s)], 
        batch_size=config['batch_size'], shuffle=True)

    # 7. Build train dataloader, and visualize - data_s
    dl_t = DataLoader(datasets_t[(train_ds_t)], 
        batch_size=config['batch_size'], shuffle=True)

    dl_test_t = DataLoader(datasets_t[(test_ds_t)], 
        batch_size=config['batch_size'], shuffle=True)

    # Segmentation model
    model_seg = UNet2D(input_shape, nr_labels, num_encoding_blocks=5)#, out_channels_first_layer=64)
    model_seg.to(device)
    agent_seg = SegmentationAgent(model=model_seg, label_names=label_names, device=device)
    agent_seg.restore_state(states_path=model_paths[config['unet_model_path']], state_name=config['unet_state_name'])

    model_seg = agent_seg.model

    # 8. Initialize model
    model = SA_Full(input_shape, nr_labels, config=config, segmentor=model_seg)
    model.to(device)



    # 9. Define loss and optimizer
    loss_g = LossDiceBCE(bce_weight=1., smooth=1., device=device)
    loss_f = LossClassWeighted(loss=loss_g, weights=config['class_weights'], 
        device=device)
    optimizer_G = model.optimizer_G
    optimizer_D = model.optimizer_D

    # 10. Train model
    results = Result(name='training_trajectory')   
    agent = SA_Agent(model=model, label_names=label_names, device=device)

    #if config['init_epoch'] > 0:
    #    agent.restore_state(exp_run.paths['states'], 'epoch_' + str(config['init_epoch']), optimizer_G, optimizer_D)

    if config['init_epoch'] > 0:
        results = Result(name='training_trajectory') 
        #results = exp_run.load_results()
        agent.restore_state(states_path=exp_run.paths['states'], state_name='epoch_' + str(config['init_epoch']), optimizer_G=optimizer_G, optimizer_D=optimizer_D)
    else: 
        results = Result(name='training_trajectory') 

    if config['style_transfer'] == False:
        agent.train(results, optimizer_G, optimizer_D, loss_f, train_dataloader_s=dl_s, train_dataloader_t=dl_t,
            init_epoch=config['init_epoch'], nr_epochs=1625, run_loss_print_interval=5,
            eval_dataloader_s=dl_test_s, eval_dataloader_t=dl_test_t, eval_interval=25,
            save_path=exp_run.paths['states'], save_interval=25)
    else:
        agent.style_transfer(results, train_dataloader_s=dl_s, train_dataloader_t=dl_t,
            eval_dataloader_s=dl_test_s, eval_dataloader_t=dl_test_t,
            save_path=exp_run.paths['states'])

    # 11. Save and print results for this experiment run
    exp_run.finish(results=results, plot_metrics=['Loss_G_Mean', 'Loss_D_Mean'])
    test_ds_key = 'test'#test_ds_key = '_'.join(dl_test_t)
    metric = 'Loss_G_Mean'
    last_dice = results.get_epoch_metric(
        results.get_max_epoch(metric, data=test_ds_key), metric, data=test_ds_key)
    print('Last Dice score for prostate class: {}'.format(last_dice))

