# Group 1.3

Topic: **Weakly Supervised Aspect Extraction Using a Student-Teacher Co-Training Approach**

--------

# Current Progress

- original edition on boots: 

  epoch: 2, f1_mid: 0.498, prev_best: 0.479 (streaming output truncated to the last 5000 lines, cannot see the result of epoch 0 and epoch 1, but in my memory loss in epoch 0 is lower?)

- original edition on bags_and_cases, 3 epochs

  loss 553.58, f1_mid 0.430; loss: 553.44, f1_mid: 0.478; loss: 553.442, f1_mid: 0.505

- change `calc_z` (sum over aspect dimensions)

  loss: 540.16, f1_mid: 0.559; loss: 540.31, f1_mid: 0.565; loss: 540.310, f1_mid: 0.565

- assign general aspect after softmax

  loss: 313.53, f1_mid: 0.533; loss: 311.80, f1_mid: 0.556; loss: 311.646, f1_mid: 0.568

# Instructions 

## training

You can open `./JupyterNotebook/debug_train.ipynb` and run some cells for training.

## Jupyter Notebooks

use whatever you like to develop code, below is the description of available jupyter notebooks

in `./JupyterNotebook`

Files with prefix `debug_` act as proxy of command line, files with prefix `draft_` are for making code clear by decomposing and printing 

- `extract_from_json.ipynb`: read all `.json` files in the given folder, extract specified content (e.g. articles and comments) and save as pickle or csv in `../processed`
- `WordEmbeddings.ipynb`: read extracted text from `../processed/`, build vocabulary and fine-tune on the corpus, save the model (word vectors) in `../wv` 
- `SeedWords.ipynb`: currently there are three approaches to find seed words: k-means on word vectors, NMF on corpus, clarity scoring function using annotated data
- `NMF.ipynb`:
- `Clarity Scoring.ipynb`
- `debug_train.ipynb`: for easy training debugging 

Folder Structure

```bash
.
├─curated-source-dataset
│  ├─english
│  └─german
├─annotated-dataset
│  ├─annotated_3rd_round
│  ├─...
│  └─...
├─JupyterNotebook	# jupyter notebook scripts
├─processed			# processed data and other useful files
├─wv				# word vectors
├─output            # output files
└─scripts           # scripts
```



