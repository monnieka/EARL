# Official Repository for EARL: Embracing Amnesic Replay for Learning with Noisy Labels

This source code is based on the [mammoth](https://github.com/aimagelab/mammoth/) framework.

## Setup

+ Use `./utils/main.py` to run experiments.
+ Use `--noise=sym` to inject synthetic noise
+ Use `--noise_type` to choose between variants of human-generated noise. Default in our exps is "noisy100"
+ Use `--noise_rate` to choose percentage of noise to load (in `[0,1]`)
+ `--buffer_size` (for methods that use it). See supplementary material for the specific size for the dataset.

### Models

Different models can be loaded with `--model=<model_name>` with `<model_name>` being:
+ `earl`: Our proposal, with amnesic replay and and bi-fold loss-aware sampling combined. Additional arguments (for ablations):
    - `--enable_amnesia` (default `1`): enable or disable amnesic replay
    - `--bifold_sampling` (default `1`): enable or disable bi-fold loss-aware sampling
+ `puridiver`: Additional arguments:
    - `--use_bn_classifier` (defult `1`): Add a final batch normalization layer after the classifier (as per the original code).
    - `--initial_alpha` (default `0.5`): Initial value of `alpha` to weight purity and diversity.
    - `--disable_train_aug` (default `0`): Disable data augmentation during train as in original `PudiDivER` (we found we get better results WITH data augmentation for the offline setting).
+ `dividermix`: Our custom `iDivideMix` baseline (requires `--lnl_mode=dividemix`)
+ `er`: Base ER model based on Reservoir sampling. Combined with buffer fitting in our experiments.

NOTE: `spr` and `cnll` are missing as are not part of the main comparison but can be found in the respective original repositories.

## e.g. to run our model on cifar100 sym noise 40% experiment
python ./utils/main.py --model=earl --dataset=seq-cifar100 --buffer_size=500 --lr=0.03 --noise=sym --noise_rate=0.4 
## e.g. to run our model on cifar100 human-annotated noise experiment
python ./utils/main.py --model=earl --dataset=seq-cifar100 --buffer_size=500 --lr=0.03 --noise_type=noisy100 


### Datasets

Can be selected with `--dataset=<dataset_name>`:
+ `seq-animal`: Sequential ANIMAL-10N
+ `seq-cifar100`: Sequential CIFAR-100
+ `seq-eurosat`: Sequential EuroSAT: Needs to be downloaded from [https://madm.dfki.de/files/sentinel/EuroSAT.zip]
+ `seq-isic`: Sequential ISIC: Needs to be downloaded from [https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Training_Input.zip]
