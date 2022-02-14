
# 1. Imports
import torch
import sys
from torch.utils.data import DataLoader
import torch.optim as optim
from mp.experiments.experiment import Experiment
from mp.data.data import Data
from mp.data.datasets.ds_hippocampus import Hippocampus
from mp.data.pytorch.pytorch_cddGan_dataset import PytorchCDDGanDataset
from mp.models.classification.resnet18 import Resnet18
from mp.eval.losses.losses_classification import CrossEntropyLoss, L1Loss
from mp.agents.resnet18_agent import Resnet18Agent
from mp.eval.losses.losses_segmentation import LossClassWeighted, LossDiceBCE
from mp.eval.result import Result

# 2. Define configuration
config = {'experiment_name':'cnn_Hippocampus_standard_b400', 'device':'cuda:0',
    'nr_runs': 1, 'cross_validation': False, 'val_ratio': 0.0, 'test_ratio': 0.3,
    'input_shape': (1, 64, 64), 'resize': False, 'augmentation': 'none', 
    'reload': False, 
    'state_name': "epoch_100",
    'class_weights': (0.,1.), 'lr': 0.0001, 'batch_size': 400, 'output_labels': 3,
    'loss_function': 'CrossEntropyLoss', # L1Loss, CrossEntropyLoss
    'dataset':'Hippocampus_all',
    'dataset2':'Hippocampus_hue',
    'evaluate': False,
    'OnlyOneDataset': False,
    'new_data_split': False,
    'domain_prefixes':['b', 'e', 's'],
    } 

if config['evaluate'] == True:
    config['batch_size'] = 1
    config['reload'] = True
    #config['device'] = 'cpu'
if config['OnlyOneDataset']:
    config['new_data_split'] = True

device = config['device']
if config['device'] != 'cpu':
    device_name = torch.cuda.get_device_name(device)
    print('Device name: {}'.format(device_name))
else: 
    print('Device name: CPU')
input_shape = config['input_shape']  

# 3. Create experiment directories
exp = Experiment(config=config, name=config['experiment_name'], notes='', reload_exp=config['reload'])

# 4. Define data
data = Data()
data.add_dataset(Hippocampus(merge_labels=True, global_name=config['dataset'], domain_prefixes=config['domain_prefixes']))
nr_labels = data.nr_labels
label_names = data.label_names
label_names= ['1', '2', '3']
train_ds = (config['dataset'], 'train')
test_ds = (config['dataset'], 'test')

# 5. Create data splits for each repetition
if config['new_data_split']:
    exp.new_data_splits(data)
else:
    exp.set_data_splits(data)

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

    if config['OnlyOneDataset'] == False:
        # 7. Build train dataloader, and visualize
        dl = DataLoader(datasets[(train_ds)], batch_size=config['batch_size'], shuffle=True)
        dl_test = DataLoader(datasets[(test_ds)], batch_size=config['batch_size'], shuffle=True)
    else: 
        # 7. Build train dataloader, and visualize
        dl = DataLoader(datasets[(train_ds)] + datasets[(test_ds)], batch_size=config['batch_size'], shuffle=True)
        dl_test = dl 

    # 8. Initialize model
    model = Resnet18(input_shape, config['output_labels'])
    model.to(device)

    # 9. Define loss and optimizer
    if config['loss_function'] == 'CrossEntropyLoss':
        loss_f = CrossEntropyLoss()
    elif config['loss_function'] == 'L1Loss':
        loss_f = L1Loss()
    else:
        print("ERROR: Loss function not implemented")
        sys.exit()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    # 10. Train model
    results = Result(name='training_trajectory')   
    agent = Resnet18Agent(model=model, label_names=label_names, device=device)

    if config['reload'] == True:
        agent.restore_state(states_path=exp_run.paths['states'], state_name=config['state_name'], optimizer=optimizer)


    agent.loss_name = config['loss_function']
    agent.output_labels = config['output_labels']
    if config['evaluate'] == False:
        agent.train(results, optimizer, loss_f, train_dataloader=dl,
            init_epoch=0, nr_epochs=100, run_loss_print_interval=0,
            eval_datasets=datasets, eval_interval=10,
            save_path=exp_run.paths['states'], save_interval=10)
    else:
        agent.evaluate(train_dataloader=dl, eval_dataloader=dl_test, loss=config['loss_function'], output_labels=config['output_labels'])

    # 11. Save and print results for this experiment run
    exp_run.finish(results=results, plot_metrics=['Mean_' + config['loss_function']])
    test_ds_key = '_'.join(test_ds)
    metric = 'Mean_' + config['loss_function']
    last_dice = results.get_epoch_metric(
        results.get_max_epoch(metric, data=test_ds_key), metric, data=test_ds_key)
    print('Last Dice score for prostate class: {}'.format(last_dice))

