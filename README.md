# HCMFDS-Net: An improved STM Network with Hybrid CNN-Mamba Encoder and Frequency Domain Decoder for CT Organs Segmentation

![model](https://github.com/user-attachments/assets/708a8d3e-37bb-480f-a5c9-5029a4078f50)


**Install with pip:**
```
pip install -r requirments.txt
```
(If you encounter the File "setup.py" not found error, upgrade your pip with pip install --upgrade pip)

## Training
**Setting Up Data**
```
The directory structure should look like this:
├── HCMFDS
├── LiTS17
│   ├── test
│   │   ├── Annotations
│   │   └── ...
│   └── trainval
│           ├── Annotations
│           └── ...
```
**Training Command**
```
OMP_NUM_THREADS=4 torchrun --master_port 25357 --nproc_per_node=4 train.py exp_id=[some unique id] model=base data=base
```

·Change nproc_per_node to change the number of GPUs.

·Prepend CUDA_VISIBLE_DEVICES=... if you want to use specific GPUs.

·Change master_port if you encounter port collision.

·exp_id is a unique experiment identifier that does not affect how the training is done.

·Models and visualizations will be saved in ./output/.

·For pre-training only, specify main_training.enabled=False.

·For main training only, specify pre_training.enabled=False.

·To load a pre-trained model, e.g., to continue main training from the final model from pre-training, specify weights=[path to the model].
