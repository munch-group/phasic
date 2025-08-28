wget https://github.com/quarto-dev/quarto-cli/releases/download/v1.7.33/quarto-1.7.33-linux-arm64.tar.gz
mkdir ~/opt
tar -C ~/opt -xvzf quarto-1.7.33-linux-arm64.tar.gz
mkdir ~/.local/bin
ln -s ~/opt/quarto-1.7.33/bin/quarto ~/.local/bin/quarto
( echo ""; echo 'export PATH=$PATH:~/.local/bin\n' ; echo "" ) >> ~/.bash_profile
source ~/.bash_profile

quarto check