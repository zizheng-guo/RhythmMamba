conda remove --name rppg-toolbox --all -y
conda create -n rppg-toolbox python=3.8 pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia