FROM mcr.microsoft.com/azureml/o16n-sample-user-base/ubuntu-miniconda

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt-get update --fix-missing && \
    apt install -y gcc python3-dev python3-pip libxml2-dev libxslt1-dev zlib1g-dev g++ && \
    apt-get install -y wget bzip2 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

#copy all code
COPY ./*.py ./
COPY ./*.ipynb ./

#copy needed files to build the image
COPY ./environment.yml ./
COPY ./data/ ./
COPY ./jupyter_run.sh ./ 


#specify the default shell 
SHELL [ "/bin/bash", "-c" ]

# make non-activate conda commands available
ENV PATH=/opt/miniconda/bin:$PATH
RUN echo ". /opt/miniconda/etc/profile.d/conda.sh" >> ~/.profile

# will get the environment.yml file by default and create the 
RUN ["conda", "env", "create"]

#entry point for environment activation on startup
ENTRYPOINT [ "./jupyter_run.sh" ]