FROM continuumio/miniconda3:latest

# Установка системных зависимостей (libsndfile1 для SoundFile)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Установка рабочей директории
WORKDIR /app

# Копирование файла environment.yml для установки зависимостей
COPY environment.yml .

# Создание conda окружения
RUN conda env create -f environment.yml && conda clean -afy

# Копирование остальных файлов проекта
COPY . .

# Загрузка и распаковка весов модели
RUN mkdir -p /app/experiments/trained_model && \
    wget https://github.com/eloimoliner/denoising-historical-recordings/releases/download/v0.0/checkpoint.zip -O checkpoint.zip && \
    unzip checkpoint.zip -d /app/experiments/trained_model/ && \
    rm checkpoint.zip

# Установка ENTRYPOINT для запуска команд внутри conda-окружения
# Команды, переданные в 'docker run', будут выполнены с помощью этой точки входа.
ENTRYPOINT ["conda", "run", "-n", "historical_denoiser", "--no-capture-output"]

# Команда по умолчанию, если пользователь не укажет свою при запуске 'docker run'.
# Например, можно запустить интерактивную сессию bash.
CMD ["/bin/bash"] 