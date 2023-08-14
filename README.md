## The Data:

50salads：https://cvip.computing.dundee.ac.uk/datasets/foodpreparation/50salads/

GTEA：https://ai.stanford.edu/~alireza/GTEA_Gaze_Website/

Breakfast：http://actionrecognition.net/files/dsetdetail.php?did=9

## Required：

- PyTorch 1.81+cu102
- Python 3.8



###  Training:

* Download the [data](https://mega.nz/#!O6wXlSTS!wcEoDT4Ctq5HRq_hV-aWeVF1_JB3cacQBQqOLjCIbc8) folder, which contains the features and the ground truth labels. (~30GB) (If you cannot download the data from the previous link, try to download it from [here](https://zenodo.org/record/3625992#.Xiv9jGhKhPY))
* To train the model run `python main.py --action=train --dataset=DS --split=SP` where `DS` is `Drilling`, `50salads` or `gtea`, and `SP` is the split number (1-5) for 50salads , (1-4) for GTEA, 1 for Drilling.

### Prediction:

Run `python main.py --action=predict --dataset=DS --split=SP`. 

### Evaluation:

Run `python eval.py --dataset=DS --split=SP`. 



