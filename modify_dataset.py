# ----------------------------------------
# Used to create synthetic domain switches
# ----------------------------------------
import torchio
import os
import random
import tqdm
import numpy as np

# Define configurations
config = {
'path_in':'in', # images in folder imagesTr, labels in folder labelsTr
'path_out':'out',
'random_seed':874653,
'full_random':True,
'folder_suffix': '',
'remove_first_x_letters':0
}
modifications = [
    {
        'enabled':False,
        'prefix':'', # brightness
        'transformations':[
            #{'type':'contrast', 'strength':1.5}
            #{'type':'contrast', 'strength':2}
            #{'type':'elastic_deformation'}
            #{'type':'brightness', 'strength':1000},
            #{'type':'affine', 'scale_range':(0.9, 1.1), 'rotation_range':(0, 20), 'translation_range':(0, 0.5)},
            {'type':'blur', 'strength':1.1},
            #{'type':'elastic_deformation', 'grid_size':7}
            #{'type':'elastic_deformation', 'grid_size':5}
            {'type':'random_noise'}
        ]
    },
    {
        'enabled':False,
        'prefix':'e', # elastic
        'transformations':[
            {'type':'elastic_deformation'}
        ]
    },
    {
        'enabled':True,
        'prefix':'s', # standard s,b,e
        'transformations':[

        ]        
    }
]

# Initializations
random.seed(config['random_seed'])
data = []
data_name = []

data_masks = []

# Load dataset
def load_data():
    data.clear()
    data_name.clear()
    path_in = os.path.join(config['path_in'], 'imagesTr')
    pbar = tqdm.tqdm(desc="Loading data", total=len([name for name in os.listdir(path_in)])) 
    for f in os.listdir(path_in):
        path_loc = os.path.join(path_in, f)
        pbar.update(1)
        if os.path.isfile(path_loc):
            data_name.append(f)
            data.append(torchio.Image(path_loc, type=torchio.INTENSITY))
    pbar.close()

def load_masks():
    data_masks.clear()
    path_out = os.path.join(config['path_in'], 'labelsTr')
    pbar = tqdm.tqdm(desc="Loading masks", total=len([name for name in os.listdir(path_out)])) 
    for f in os.listdir(path_out):
        path_loc = os.path.join(path_out, f)
        pbar.update(1)
        if os.path.isfile(path_loc):
            #data_name.append(f)
            data_masks.append(torchio.Image(path_loc, type=torchio.INTENSITY))
    pbar.close()
# Transform data according to specified modifications
def transformations(mods):
    for m in mods:
        if m['type'] == 'brightness':
            brightness_change(m)
        if m['type'] == 'affine':
            affine(m)
        if m['type'] == 'elastic_deformation':
            elastic_deformation(m)
        if m['type'] == 'blur':
            blur(m)
        if m['type'] == 'random_noise':
            noise(m)
        if m['type'] == 'random_ghosting':
            random_ghosting(m)
        if m['type'] == 'random_spike':
            random_spike(m)
        if m['type'] == 'contrast':
            contrast(m)
        if m['type'] == 'random_swap':
            random_swap(m)
        if m['type'] == 'random_bias_field':
            random_bias_field(m)

# Random bias field added to image
def random_bias_field(mod):
    pbar = tqdm.tqdm(desc="Random bias field", total=len(data))
    for d in data:
        transform = torchio.RandomBiasField()
        d.data = transform(d.data)
        pbar.update(1)
    pbar.close()  

# Random swap added to image
def random_swap(mod):
    pbar = tqdm.tqdm(desc="Random swap", total=len(data))
    for d in data:
        transform = torchio.RandomSwap(patch_size=(6,6,1))
        d.data = transform(d.data)
        pbar.update(1)
    pbar.close()  

# Simple contrast change of whole dataset
def contrast(mod):
    strength = mod['strength']
    pbar = tqdm.tqdm(desc="Contrast transformation", total=len(data))
    for d in data:
        strength = float(random.randint(100, int(mod['strength']*100)))/100        
        d.data = np.power(d.data, strength) #d.data * strength#
        pbar.update(1)
    pbar.close()

# Random ghosting added to image
def random_ghosting(mod):
    pbar = tqdm.tqdm(desc="Random ghosting", total=len(data))
    for d in data:
        transform = torchio.RandomGhosting()
        d.data = transform(d.data)
        pbar.update(1)
    pbar.close()       

# Random spike added to image
def random_spike(mod):
    pbar = tqdm.tqdm(desc="Random spike", total=len(data))
    for d in data:
        transform = torchio.RandomSpike()
        d.data = transform(d.data)
        pbar.update(1)
    pbar.close()      

# Random noise added to image
def noise(mod):
    pbar = tqdm.tqdm(desc="Random noise", total=len(data))
    for d in data:
        transform = torchio.RandomNoise(mean=0.015, std=0.0075)
        d.data = transform(d.data)
        pbar.update(1)
    pbar.close()       

# Simple brightness change of whole dataset
def brightness_change(mod):
    strength = mod['strength']
    pbar = tqdm.tqdm(desc="Brightness transformation", total=len(data))
    for d in data:
        if config['full_random']:
            strength = random.randint(mod['strength']/2, mod['strength'])        
        d.data = d.data + strength
        pbar.update(1)
    pbar.close()

# Random affine transformation in the range of the set parameters
def affine(mod):
    if config['full_random']:
        transform = torchio.RandomAffine()
    else:
        scales = tuple(random.uniform(*mod['scale_range']) for _ in range(3))
        degrees = tuple(random.uniform(*mod['rotation_range']) for _ in range(3))
        translation = tuple(random.uniform(*mod['translation_range']) for _ in range(3))
        transform = torchio.Affine(scales=scales, degrees=degrees, translation=translation)
    pbar = tqdm.tqdm(desc="Affine transformation", total=len(data))
    for d in data:
        d.data = transform(d.data)
        pbar.update(1)
    pbar.close()

def blur(mod):
    if config['full_random']:
        transform = torchio.RandomBlur()
    else:
        transform = torchio.Blur(mod['strength'])
    pbar = tqdm.tqdm(desc="Blur transformation", total=len(data))
    for d in data:
        d.data = transform(d.data)
        pbar.update(1)
    pbar.close()


def generate_control_points(size):
    control_points = []
    factor = 1/size
    for x in range(size+1):
        for y in range(size+1):
            control_points.append(np.array([[x*factor], [y*factor], [0]]))
    return np.array([control_points])

def elastic_deformation(mod):
    pbar = tqdm.tqdm(desc="Elastic transformation", total=len(data))
    for d in data:
        transform = torchio.RandomElasticDeformation(locked_borders=2, max_displacement=25) #12,25,
        d.data = transform(d.data)
        pbar.update(1)
    pbar.close()

def create_output_dirs():
    os.makedirs(os.path.join(config['path_out'], 'imagesTr' + config['folder_suffix']), exist_ok=True)
    os.makedirs(os.path.join(config['path_out'], 'labelsTr' + config['folder_suffix']), exist_ok=True)

# Save resulting dataset
def save_data(prefix):
    pbar = tqdm.tqdm(desc="Save data", total=len(data))
    os.makedirs(config['path_out'], exist_ok=True)
    for idx, d in enumerate(data):
        out_data = torchio.Image(tensor=d.data, affine=d.affine)
        out_data.save(os.path.join(config['path_out'], 'imagesTr' + config['folder_suffix'], prefix + data_name[idx][config['remove_first_x_letters']:]))
        pbar.update(1)
    pbar.close()
    pbar = tqdm.tqdm(desc="Save masks", total=len(data))
    for idx, d in enumerate(data_masks):
        out_data = torchio.Image(tensor=d.data, affine=d.affine)
        out_data.save(os.path.join(config['path_out'], 'labelsTr' + config['folder_suffix'], prefix + data_name[idx][config['remove_first_x_letters']:]))
        pbar.update(1)
    pbar.close()


if __name__ == "__main__":
    for i, d in enumerate(modifications):
        if modifications[i]['enabled'] is True:
            print('----- Process Dataset ' + str(i+1) + '/' + str(modifications.__len__()) + ' -----')
            print('Transformations:')
            print(d['transformations'])
            load_data()
            load_masks()
            transformations(d['transformations'])
            create_output_dirs()
            save_data(d['prefix'])
    print('PROCESSING COMPLETE!')