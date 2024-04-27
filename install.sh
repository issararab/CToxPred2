conda config --add channels defaults
conda config --add channels bioconda
conda config --add channels conda-forge
conda install -y -c rdkit -c mordred-descriptor mordred
conda install -y -c conda-forge pybel
conda install -y -c openbabel openbabel=3.1.1
pip install numpy==1.23.5
pip install scikit-learn==1.3.1
pip install scipy==1.11.4
pip install MolVS==0.1.1
pip install pandas==2.0.3
pip install torch==1.12.1
pip install torch-geometric==2.3.1
pip install git+https://github.com/gadsbyfly/PyBioMed