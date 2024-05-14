mamba env create -f environment.yml
mamba activate mamba-4chan
mamba install cuda-toolkit -c nvidia -y # when install with environment.yml, cuda sometimes explodes.
git clone --branch v1.1.3.post1 https://github.com/Dao-AILab/causal-conv1d
pip install -e causal-conv1d/
git clone --branch v1.1.4 https://github.com/state-spaces/mamba
pip install -e mamba/