# Content Domain Disentanglement GAN (CDD GAN)
The code to our paper 'DISENTANGLEMENT ENABLES CROSS-DOMAIN HIPPOCAMPUS SEGMENTATION' from John Kalkhof, Camila Gonzalez and Anirban Mukhopadhyay. 

## Quickstart
After installation follow these steps to execute CDD-GAN.

### Image preparation
- Images need to be in the '.nii.gz' format and already seperated into slices. 
- Folder structure:
    - Dataset/
        - imagesTr/
        - labelsTr/ 
- Image naming scheme:
    - XY*_ID_SLICE
    - X = Single letter prefix
    - Y* = Can be anything (only for readability)
    - ID = id of patient
    - SLICE = slice number

### Paths
- Open paths.py
- Define your dataset paths here
- Define pretrained UNet path here if you want to use CDD-GAN_M

### CDD-GAN Configuration
- Open CDD-GAN_training.py
- Edit: dataset, input_shape, domain_prefixes, number_domain, number_identity
- Other variables are optional

## Install with Anaconda:
0. (Create a Python3.8 environment, e.g. as conda create -n <env_name> python=3.8, and activate)
2. Install CUDA if not already done and install PyTorch through conda with the command specified by https://pytorch.org/. The tutorial was written using PyTorch 1.6.0. and CUDA10.2., and the command for Linux was at the time 'conda install pytorch torchvision cudatoolkit=10.2 -c pytorch'
3. cd to the project root (where setup.py lives)
4. Execute 'pip install -r requirements.txt'
5. Set paths in mp.paths.py
6. Execute git update-index --assume-unchanged mp/paths.py so that changes in the paths file are not tracked
7. Execute 'pytest' to test the correct installation. Note that one of the tests will test whether at least one GPU is present, if you do not wish to test this mark to ignore. The same holds for tests that used datasets which much be previously downloaded.