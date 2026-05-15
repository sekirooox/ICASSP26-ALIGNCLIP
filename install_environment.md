## Installation

```bash
conda create --name alignclip python=3.7 -y
conda activate alignclip
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
pip install openmim
mim install mmcv-full==1.5.0
pip install mmsegmentation==0.24.0
pip install -r requirements.txt
```
