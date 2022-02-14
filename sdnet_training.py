# 1. Imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from mp.experiments.experiment import Experiment
from mp.data.data import Data
from mp.data.datasets.ds_hippocampus import Hippocampus
from mp.data.pytorch.pytorch_seg_dataset import PytorchSeg2DDataset
from mp.data.pytorch.pytorch_cddGan_dataset import PytorchCDDGanDataset
from mp.models.sdnet.sdnet import SD_Net
from mp.eval.losses.losses_segmentation import LossClassWeighted, LossDiceBCE
from mp.agents.sdnet_agent import SDNetAgent
from mp.eval.result import Result
from datetime import datetime

print(datetime.now())

# 2. Define configuration
config = {'experiment_name':'sdnet_train', 'device':'cuda:0',
    'nr_runs': 1, 'cross_validation': False, 'val_ratio': 0.0, 'test_ratio': 0.3,
    'input_shape': (1, 64, 64), 'resize': False, 'augmentation': 'none', 
    'class_weights': (0.,1.), 'lr': 0.0001, 'batch_size': 2, 'reload': False, 'state_name': "epoch_260",
    'dataset':'Hippocampus_standard',
    'style_transfer': False,
    'new_datasplit':False,
    'domain_prefixes':['b', 'e', 's'],
    }
device = config['device']
device_name = torch.cuda.get_device_name(device)
print('Device name: {}'.format(device_name))
input_shape = config['input_shape']  

# 3. Create experiment directories
exp = Experiment(config=config, name=config['experiment_name'], notes='', reload_exp=True)

# 4. Define data
data = Data()
data.add_dataset(Hippocampus(merge_labels=True, global_name=config['dataset'], domain_prefixes=config['domain_prefixes']))
nr_labels = data.nr_labels
label_names = data.label_names
train_ds = (config['dataset'], 'train')
test_ds = (config['dataset'], 'test')

# 5. Create data splits for each repetition
if not config['new_datasplit']:
    exp.set_data_splits(data)
else:
    exp.new_data_splits(data)

# Now repeat for each repetition
for run_ix in range(config['nr_runs']):
    exp_run = exp.get_run(run_ix=run_ix, reload_exp_run=config['reload'])

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

    dict_test = {
        "train": dl,
        "test": dl_test
    }


    # 8. Initialize model
    model = SD_Net(input_shape, nr_labels)
    model.to(device)

    # 9. Define loss and optimizer
    loss_g = LossDiceBCE(bce_weight=1., smooth=1., device=device)
    loss_f = LossClassWeighted(loss=loss_g, weights=config['class_weights'], 
        device=device)

    loss_MAE = nn.L1Loss()
    optimizer_sdnet = model.sdnet_optimizer
    optimizer_sdnet_z = model.sdnet_optimizer_z


    # 10. Train model
      
    agent = SDNetAgent(model=model, label_names=label_names, device=device)
    if config['reload']:
        results = exp_run.load_results()
        agent.restore_state(states_path=exp_run.paths['states'], state_name=config['state_name'])
    else: 
        results = Result(name='training_trajectory') 
    if config['style_transfer'] is False:
        agent.train(results, optimizer_sdnet, optimizer_sdnet_z, loss_f, loss_MAE, train_dataloader=dl,
            init_epoch=0, nr_epochs=200, run_loss_print_interval=1,
            eval_datasets=dict_test, eval_interval=10,
            save_path=exp_run.paths['states'], save_interval=10)
    else: 
        agent.style_transfer(results, optimizer_sdnet, optimizer_sdnet_z, loss_f, loss_MAE, train_dataloader=dl, eval_datasets=dict_test, load_domain_image_path="M:/MasterThesis/Datasets/Hippocampus/preprocessed_dataset_train/imagesTr/hippocampus_001_0.nii.gz",save_path=exp_run.paths['states'])

    # 11. Save and print results for this experiment run
    exp_run.finish(results=results, plot_metrics=['Mean_Dice', 'Mean_Divergence', 'Mean_Loss_Rec', 'Mean_Loss_Z'])# 
    test_ds_key = '_'.join(test_ds)
    metric = 'Mean_ScoreDice[prostate]'
    last_dice = results.get_epoch_metric(
        results.get_max_epoch(metric, data=test_ds_key), metric, data=test_ds_key)
    print('Last Dice score for prostate class: {}'.format(last_dice))

print(datetime.now())