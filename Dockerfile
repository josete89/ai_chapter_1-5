FROM continuumio/anaconda3

RUN mkdir app

WORKDIR app

COPY enviroment.yml .

COPY notebook .

RUN conda env create -f enviroment.yml

EXPOSE 8888

ENV env_name=Python3

ENTRYPOINT ["/bin/bash","-c","source activate ai_chapter_1,5 && python -m ipykernel install --user --name $env_name  --display-name $env_name   && jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root"] 
