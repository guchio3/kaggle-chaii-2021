#FROM gcr.io/kaggle-images/python:latest
FROM gcr.io/kaggle-gpu-images/python

# ENV vars
ARG WANDB_API_KEY

# set env
ENV LC_ALL "en_US.UTF-8"

# set dotfiles
WORKDIR /root
RUN git clone https://github.com/guchio3/guchio_utils.git
RUN rm .bashrc && ln -s /root/guchio_utils/.bashrc .bashrc
# RUN ln -s /root/guchio_utils/pudb/pudb.cfg .config/pudb/pudb.cfg

# dev env for GLIBCXX_3.4.26
RUN add-apt-repository ppa:ubuntu-toolchain-r/test -y
RUN apt-get update -y
# RUN apt-get install gcc-4.9 -y
RUN apt-get upgrade libstdc++6 -y

# install git-completion.bash and git-prompt.sh
RUN wget https://raw.githubusercontent.com/git/git/master/contrib/completion/git-completion.bash -O ~/.git-completion.bash \
    && chmod a+x ~/.git-completion.bash
RUN wget https://raw.githubusercontent.com/git/git/master/contrib/completion/git-prompt.sh -O ~/.git-prompt.sh \
    && chmod a+x ~/.git-prompt.sh

# install conda packages
RUN conda install -y -c conda-forge pudb

# install pip packages
RUN pip install japanize_matplotlib wandb

RUN wandb login $WANDB_API_KEY

# set jupyter notebook
# jupyter vim key-bind settings
RUN pip install jupyter_contrib_nbextensions
RUN jupyter contrib nbextension install --user
RUN mkdir -p $(jupyter --data-dir)/nbextensions
RUN git clone https://github.com/lambdalisue/jupyter-vim-binding $(jupyter --data-dir)/nbextensions/vim_binding
RUN jupyter nbextension enable vim_binding/vim_binding
# edit vim_bindings setting as I can use C-c for exitting insert mode
RUN sed -i "s/      'Ctrl-C': false,  \/\/ To enable clipboard copy/\/\/      'Ctrl-C': false,  \/\/ To enable clipboard copy/g" $(jupyter --data-dir)/nbextensions/vim_binding/vim_binding.js
