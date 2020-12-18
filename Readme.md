## The improvement of Super-Resolution Convolutional Networks using FSRCNN
This repository re-implement the SRCNN, FSRCNN  and several related model for comparison purpose.
### Paper resources: 
- SRCNN: https://arxiv.org/abs/1501.00092
- FSRCNN: https://arxiv.org/abs/1608.00367


### Model List
- SRCNN
- SRCNN_WO_1: SRCNN model without first layer
- SRCNN_WO_2: SRCNN model without second layer
- FSRCNN
- FSRCNN_S1: SRCNN with deconv layer as last later 
- FSRCNN_S2: SRCNN with deconv layer as last later and 6 mid layer from FSRCNN as mid layer

### Command To execute
####Train:
>python main.py --config-file config.ini

#### Test:
##### Generate the test output:
>python test.py --config-file testConfigs/CONFIG_FILE_NAME.ini
##### Get the model average pred time by evalDataset:
>python test.py --config-file testConfigs/CONFIG_FILE_NAME.ini evalDataset/DATASET_NAME.h5
