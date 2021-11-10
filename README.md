## File structure
* cfgs/brain  - directory with configurations for models in each experiment
* datasets, function, ops, utils  - directories with all the tools used in the project
* models  - directory with models used in the project
* Brain_to_point_clouds.ipynb - transforms 3d tensors of brains into Point Clouds and saves it both for FCD detection task and for grey matter segmentation.
* Experiment*.ipynb - training the model
* Inference.ipynb - use models for validation
* Result_viewer.ipynb - create tables with metrics (used for report)

------

Add data folder "croped_new_dataset" with two folders in it: 
1. "fcd_brains"
2. "masks"

Create folder "data" with two folders in it:
1. Empty folder "BrainData"
2. "HCP_1200" with HCP dataset

-------

For training use Experiments*.ipynb notebooks

For inference use Inference.ipynb notebook 

For evaluation use Result_viewer.ipynb notebook  
