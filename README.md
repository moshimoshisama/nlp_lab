# Group 1.3

Topic: **Weakly Supervised Aspect Extraction Using a Student-Teacher Co-Training Approach**

--------

# Instructions 

use whatever you like to develop code, below is the description of available jupyter notebooks

in `./JupyterNotebook`

- `preprocess.ipynb`: read all `.json` files in the given folder, extract specified content (e.g. articles and comments) and save as pickle or csv in `../processed`
- `WordEmbeddings.ipynb`: read extracted text from `../processed/`, build vocabulary and fine-tune on the corpus, save the model (word vectors) in `../wv` 
- `SeedWords.ipynb`: currently there are three approaches to find seed words: k-means on word vectors, NMF on corpus, clarity scoring function using annotated data
- `NMF.ipynb`:
- `Clarity Scoring.ipynb`

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
├─processed			# processed data
├─wv				# word vectors
└─output            # output files
```



