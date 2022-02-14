# 1. Imports
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from mp.experiments.experiment import Experiment
from mp.data.data import Data
from mp.data.datasets.ds_hippocampus import Hippocampus
from mp.data.pytorch.pytorch_seg_dataset import PytorchSeg2DDataset
from mp.models.segmentation.unet_fepegar import UNet2D
from mp.eval.losses.losses_segmentation import LossClassWeighted, LossDiceBCE
from mp.agents.agent_unet_eval import UnetEvalAgent
from mp.agents.segmentation_agent import SegmentationAgent
from mp.eval.result import Result

# 2. Define configuration
config = {'experiment_name':'unet_hipp_eval', 'device':'cuda:0', 
    'nr_runs': 1, 'cross_validation': False, 'val_ratio': 0.0, 'test_ratio': 0.3,
    'input_shape': (1, 64, 64), 'resize': False, 'augmentation': 'none', 
    'class_weights': (0.,1.), 'lr': 0.0001, 'batch_size': 1, 
    'reload': True, 
    'state_name': "epoch_200",
    'new_data_split': False,
    'dataset':'Hippocampus',
    'domain_prefixes':['b', 'e', 's'],
    } 
device = config['device']
if config['device'] != 'cpu':
    device_name = torch.cuda.get_device_name(device)
    print('Device name: {}'.format(device_name))
else: 
    print('Device name: CPU')
input_shape = config['input_shape']  

# 3. Create experiment directories
exp = Experiment(config=config, name=config['experiment_name'], notes='', reload_exp=False)

# 4. Define data
data = Data()
data.add_dataset(Hippocampus(merge_labels=True, global_name=config['dataset'], domain_prefixes=config['domain_prefixes']))
nr_labels = data.nr_labels
print(nr_labels)
label_names = data.label_names
train_ds = (config['dataset'], 'train')
test_ds = (config['dataset'], 'test')

# 5. Create data splits for each repetition
if config['new_data_split']:
    exp.new_data_splits(data)
else:
    exp.set_data_splits(data)

# Now repeat for each repetition
for run_ix in range(config['nr_runs']):
    exp_run = exp.get_run(run_ix=run_ix, reload_exp_run=True)

    # 6. Bring data to Pytorch format
    datasets = dict()
    for ds_name, ds in data.datasets.items():
        for split, data_ixs in exp.splits[ds_name][exp_run.run_ix].items():
            if len(data_ixs) > 0:  # Sometimes val indexes may be an empty list
                aug = config['augmentation'] if not('test' in split) else 'none'
                datasets[(ds_name, split)] = PytorchSeg2DDataset(ds, 
                    ix_lst=data_ixs, size=input_shape, aug_key=aug, 
                    resize=config['resize'])


    # 7. Build train dataloader, and visualize
    dl = DataLoader(datasets[(train_ds)], batch_size=config['batch_size'], shuffle=True)
    dl_test = DataLoader(datasets[(test_ds)], batch_size=config['batch_size'], shuffle=True)

    # 8. Initialize model
    model = UNet2D(input_shape, nr_labels, num_encoding_blocks=5)
    model.to(device)

    # 9. Define loss and optimizer
    loss_g = LossDiceBCE(bce_weight=1., smooth=1., device=device)
    loss_f = LossClassWeighted(loss=loss_g, weights=config['class_weights'], 
        device=device)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    agent = SegmentationAgent(model=model, label_names=label_names, device=device)
    # 10. Train model
    results = Result(name='training_trajectory') 
    agent.restore_state(states_path=exp_run.paths['states'], state_name=config['state_name'], optimizer=optimizer)

    agent.track_metrics(1454, results, loss_g, datasets)

    # 11. Save and print results for this experiment run
    exp_run.finish(results=results, plot_metrics=['Mean_ScoreDice', 'Mean_ScoreDice[hippocampus]', 'Mean_ScoreDice[background]'])
    test_ds_key = '_'.join(test_ds)
    metric = 'Mean_ScoreDice[hippocampus]'
    last_dice = results.get_epoch_metric(
        results.get_max_epoch(metric, data=test_ds_key), metric, data=test_ds_key)
    print('Last Dice score for prostate class: {}'.format(last_dice))

