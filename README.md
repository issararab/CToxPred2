# CToxPred2
Comprehensive cardiotoxicity prediction tool of small molecules on three targets: hERG, Nav1.5, Cav1.2


<p align="center">
	<img src="img/GUI.png" />
</p>


:exclamation:Clone first the whole repository package and follow the steps bellow.

## Prerequisites
1- Create and activate a conda environment:

		$conda create -n ctoxpred2 python=3.9
		$conda activate ctoxpred2

2- Install packages:

		$bash install.sh

3- Clone the repository: 

		$git clone git@github.com:issararab/CToxPred2.git

4- Move to the repository:

		$cd CToxPred2

5- Run test:

		$python app.py
  
The software saves the predictions to a CSV file named 'predictions.csv'

## Manuscript

https://www.biorxiv.org/content/10.1101/2023.08.15.553429v1

## Data availability

To re-train the models, re-evaluate the models using the same test sets, or re-run the analysis notebook, you will find all the data in the folder './data'.
To get the full library store comprising a whole set of ChEMBL database small molecules along with feature reprsentations, fetch the database deposited for public use on Zenodo (https://zenodo.org/records/11066707).

## Hot stuff

- Evaluation of the CToxPred performance compared toCardioGenAI, trained and tested using the same data, where (a) is performance on Test-70 and (b) performance on Test-60. 
<p align="center">
	<img src="notebooks/figures/CToxpred_vs_CardioGenAI.png" />
</p>

- t-SNE visualizations showing the distributions of the labeled and unlabeled molecules in the development set and the two external test sets (Eval-60 and Eval-70) for (a) hERG (b) Nav1.5 and (c) Cav1.2.
<p align="center">
	<img src="notebooks/figures/TSNE.png" />
</p>

