FROM continuumio/miniconda3:latest

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY environment.yml .

RUN conda env create -f environment.yml && conda clean -afy

COPY . .

RUN mkdir -p /app/experiments/trained_model && \
    wget https://github.com/eloimoliner/denoising-historical-recordings/releases/download/v0.0/checkpoint.zip -O checkpoint.zip && \
    unzip checkpoint.zip -d /app/experiments/trained_model/ && \
    rm checkpoint.zip

# Установка ENTRYPOINT для запуска команд внутри conda-окружения
# Команды, переданные в 'docker run', будут выполнены с помощью этой точки входа.
ENTRYPOINT ["conda", "run", "-n", "historical_denoiser", "--no-capture-output"]

CMD ["/bin/bash"] 