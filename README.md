# LanguageModelExperimentation

Research conducted under Prof. Kurt Keutzer at Berkeley Artificial Intelligence Research (BAIR). 
This codebase is forked from Brian Yu (https://github.com/bri25yu/LanguageModelExperimentation)  in order to accomodate a number of Sanskrit-specific tasks.

<img src="http://bair.berkeley.edu/images/BAIR_Logo_BlueType_Tag.png" width="525" height="280">



### Example setup
```bash
# Install conda if not already installed
wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Linux-x86_64.sh

conda env create -f environment.yml
conda activate sktlme

deepspeed run.py

python scripts/read_results.py
```


