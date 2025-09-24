# Illumination-Guided-Frequency-Adaptive-Network




### The overall framework of the IG-FAN
![](https://github.com/AXNing/Illumination-Guided-Frequency-Adaptive-Network/blob/main/framework.jpg)

### Installation
1. Environment
CUDA 11.8; Python 3.8; torch 1.11
2. Clone the repo

    ```bash
    git clone https://github.com/AXNing/Illumination-Guided-Frequency-Adaptive-Network.git
    ```

3. Install dependent packages

    ```bash
    cd IG_FAN
    pip install -r requirements.txt
    ```




### Data Download

Download the Flare7k++ dataset [here](https://github.com/ykdai/Flare7K).

#### Flare7k++ structure

```
├── Flare7K
    ├── Reflective_Flare 
    ├── Scattering_Flare
         ├── Compound_Flare
         ├── Glare_with_shimmer
         ├── Core
         ├── Light_Source
         ├── Streak
├── Flare-R
	├── Compound_Flare
	├── Light_Source
├── test_data
     ├── real
          ├── input
          ├── gt
          ├── mask
     ├── synthetic
          ├── input
          ├── gt
	  ├── mask

```



### Training model



Download pretrained decomposition model weights [here](https://drive.google.com/drive/folders/1a02LFXPxZBIfj0ffO4nJivn_uWuycX2R?usp=drive_link).

Add weight path in file options/IG_FAN_option.yml

Please use:

```
python basicsr/train.py -opt options/IG_FAN_option.yml
```
To train a model with your own data/model, you can edit the `options/IG_FAN_option.yml` 



### Inference Code

You can download our trained model weights [here](https://drive.google.com/drive/folders/1a02LFXPxZBIfj0ffO4nJivn_uWuycX2R?usp=drive_link).

To estimate the flare-free images with the checkpoint pretrained on Flare7K++ dataset, you can run the `test.sh` by using:

```
python test.py --input ./datasets/Flare7Kpp/test_data/real/input --output ./results --model_path ./.pth --flare7kpp
python evaluate.py --input ./results --gt ./datasets/Flare7Kpp/test_data/real/gt --mask ./datasets/Flare7Kpp/test_data/real/mask
```

### Datasets
Flare7k++ real and synthetic nighttime flare-corrupted: [link](https://github.com/ykdai/Flare7K). 

Real-world nighttime flare-corrupted dataset: [link](https://github.com/ykdai/Flare7K).

Consumer electronics test datasets: [link](https://drive.google.com/drive/folders/1J1fw1BggOP-L1zxF7NV0pYhvuZQsmiWY).





### License

This project is licensed under <a rel="license" href="https://github.com/ykdai/Flare7K/blob/main/LICENSE">S-Lab License 1.0</a>. Redistribution and use of the dataset and code for non-commercial purposes should follow this license.
