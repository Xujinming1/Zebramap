## ZebraMap

### Setup

1. **Use conda to create a new environment**
     - run 
     ```bash
     conda create -n zebra python=3.9 -y
     ```
     - You can change the environment's name if you want.
2. **Install pytorch and lightning**
     - Refer to the [PyTorch installation guide](https://pytorch.org/get-started/previous-versions/) for commands tailored to your environment.
     - we use pytorch and lightning with version 2.5.0 for cuda 11.8

3. **Install python dependencies**
     - From the root directory, run:
     ```bash
     pip install -r requirements.txt
     ```
     - We don't pinned the accurate versions of packages.


### API in config.json
We have two config.json in `train` and `test` folder. They are used for train and test, respectively. The meaning of attributes are as followed:

- **`*_data_path` (str)**
Path to the root folder of your dataset. `*` contains `[train, val, test]`, is used for train, validation and test, respectively.

- **`*_filenames_path` (str)**
Path to the `.txt` file which contains pair of names of images, depths and stripes. `*` contains `[train, val, test]`, is used for train, validation and test, respectively.

- **`normal_path` (str)**
Path to the folder containing the corresponding normal maps.

- **`stripes_path` (str)**
Path to a `.npy` file which contains the stripe patterns.

- **`depth_factor` (int)**
Divides the raw depth values to convert them into meters. For NYU Depth V2, `depth_factor=1000.0`;

- **`train_from_scratch` (bool)**  
Set  `True` when training and `False` when testing.

- **`no_of_classes` (int)**  
Number of scene classes for internal embeddings. For NYU (indoor model) use `100`;

- **`max_depth` (float)**  
Maximum depth value the model will predict. `10.0` for indoor (NYU).

-**`ckpt_path` (str)**
Path to the checkpoint file used for initializing the model.

### Train and Test
- If you want to train (test), go into folder `train (test)`.
- run 
```bash
python train.py
```
or 
```bash
python test.py
```