# Facebook Hateful Memes Challenge - Team Kingsterdam

This is the code from team Kingsterdam for the Hateful Memes Challenge by Facebook AI.

### Installation

- Create a virtual environment with Python 3.7.5 using either `virtualenv` or `conda`.
- Activate the virtual environment.
- Install the required packages using `pip install -r requirements.txt`. It includes PyTorch for CUDA version 10.1.
- Install Nvidia Apex as follows:
```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

### Clone the repository

Clone the repository using `git clone git@github.com:Nithin-Holla/meme_challenge.git`.

### Download the pretrained model

- Navigate to the directory: `cd meme_challenge`.
- The pretrained UNITER-base model can be downloaded using `wget 'https://convaisharables.blob.core.windows.net/uniter/pretrained/uniter-base.pt'`.
- Next, convert the model's state_dict to work with the code using the following snippet:
```python
import torch
model_name = 'uniter-base.pt'
checkpoint = torch.load(model_name)
state_dict = {'model_state_dict': checkpoint}
torch.save(state_dict, model_name)

```

### Obtain image features

- Make a new directory: `mkdir data`.
- Copy the HatefulMemes dataset to the `data` directory.
- Extract image features as detailed [here](https://github.com/Nithin-Holla/meme_challenge/blob/master/bottom-up-attention.pytorch/README.md) into `data/own_features`.
- Alternatively, we provide the extracted features [here](https://drive.google.com/file/d/1R6ilAEmzH0gKfZCdbiSDmFj80PFxQh7R/view?usp=sharing).

### Training

The directory structure is assumed to be as follows:
<pre>
.
├── meme_challenge/
├── data
│   ├── img/
│   ├── own_features/
│   ├── train.jsonl
│   ├── dev_seen.jsonl
│   ├── dev_unseen.jsonl
│   ├── test_seen.jsonl
│   ├── test_unseen.jsonl
</pre>

To train the model from the root of the aforementioned directory structure, run the following:
```bash
python -u train_uniter.py --config meme_challenge/config/uniter-base.json --data_path data/ --model_path meme_challenge/ --pretrained_model_file uniter-base.pt --feature_path data/own_features/ --lr 3e-5 --scheduler warmup_cosine --warmup_steps 500 --max_epoch 30 --batch_size 16 --patience 5 --gradient_accumulation 2 --confounder_repeat 3 --pos_wt 1.8 --model_save_name meme.pt --seed 43 --num_folds -1 --crossval_dev_size 200 --crossval_use_dev
```
The results will be exported as CSV files in the `meme_challenge` directory. The results from the individual folds would be named `meme_fold_1*.csv` to `meme_fold_14*.csv`. The ensemble test predictions are named `meme_test_seen_ensemble.csv` and `meme_test_unseen_ensemble.csv`.

### Inference

We provide the trained models, one each from the 15 folds. To run inference, run the following:
```bash
python -u train_uniter.py --config meme_challenge/config/uniter-base.json --data_path data/ --model_path meme_challenge/ --feature_path data/own_features/ --lr 3e-5 --scheduler warmup_cosine --warmup_steps 500 --max_epoch 0 --batch_size 16 --patience 5 --gradient_accumulation 2 --confounder_repeat 3 --pos_wt 1.8 --model_save_name meme.pt --seed 43 --num_folds -1 --crossval_dev_size 200 --crossval_use_dev
```

Our trained models and CSV files corresponding to the results are available [here](https://drive.google.com/file/d/1QIQ1GJUxDT-OTo_lbMSnGN2c28PLU_C1/view?usp=sharing). The models are named `reproduce_fold_*.pt`. To run inference with these models, run the aforementioned command by replacing `meme.pt` with `reproduce.pt`.

### Citation
If you use this code, please consider citing the paper:
```bib
@article{lippe2020multimodal,
  title={A Multimodal Framework for the Detection of Hateful Memes},
  author={Lippe, Phillip and Holla, Nithin and Chandra, Shantanu and Rajamanickam, Santhosh and Antoniou, Georgios and Shutova, Ekaterina and Yannakoudakis, Helen},
  journal={arXiv preprint arXiv:2012.12871},
  year={2020}
}
```
