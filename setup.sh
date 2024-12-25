conda remove --name RhythmMamba --all -y
conda create -n RhythmMamba python=3.8 pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia