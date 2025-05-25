# A two-stage U-Net for high-fidelity denoising of historical recordings

Official repository of the paper:

> E. Moliner and V. Välimäki,, "A two-stage U-Net for high-fidelity denosing of historical recordings", submitted to IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP), Singapore, May, 2022

## Abstract
Enhancing the sound quality of historical music recordings is a long-standing problem. This paper presents a novel denoising method based on a fully-convolutional deep neural network. A two-stage U-Net model architecture is designed to model and suppress the degradations with high fidelity. The method processes the time-frequency representation of audio, and is trained using realistic noisy data to jointly remove hiss, clicks, thumps, and other common additive disturbances from old analog discs. The proposed model outperforms previous methods in both objective and subjective metrics. The results of a formal blind listening test show that the method can denoise real gramophone recordings with an excellent quality. This study shows the importance of realistic training data and the power of deep learning in audio restoration.

<p align="center">
<img src="https://user-images.githubusercontent.com/64018465/131505025-e4530f55-fe5d-4bf4-ae64-cc9a502e5874.png" alt="Schema represention"
width="400px"></p>

Listen to our [audio samples](http://research.spa.aalto.fi/publications/papers/icassp22-denoising/)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eloimoliner/denoising-historical-recordings/blob/master/colab/demo.ipynb)

## Requirements
You will need at least python 3.7 and CUDA 10.1 if you want to use GPU. See `requirements.txt` for the required package versions.

To install the environment through anaconda, follow the instructions:

    conda env update -f environment.yml
    conda activate historical_denoiser

## Denoising Recordings

You can denoise your recordings in the cloud using the Colab notebook. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eloimoliner/denoising-historical-recordings/blob/master/colab/demo.ipynb)

Otherwise, run the following commands to clone the repository and install the pretrained weights of the two-stage U-Net model:

    git clone https://github.com/eloimoliner/denoising-historical-recordings.git
    cd denoising-historical-recordings
    wget https://github.com/eloimoliner/denoising-historical-recordings/releases/download/v0.0/checkpoint.zip
    unzip checkpoint.zip -d /experiments/trained_model/
    
If the environment is installed correctly, you can denoise an audio file by running:

    bash inference.sh "file name"
    
A ".wav" file with the denoised version, as well as the residual noise and the original signal in "mono", will be generated in the same directory as the input file.
## Training
To retrain the model, follow the instructions:

Download the [Gramophone Noise Dataset](http://research.spa.aalto.fi/publications/papers/icassp22-denoising/media/datasets/Gramophone_Record_Noise_Dataset.zip), or any other dataset containing recording noises.

Prepare a dataset of clean music (e.g. [MusicNet](https://zenodo.org/record/5120004#.YnN-96IzbmE))


## Remarks

The trained model is specialized in denoising gramophone recordings, such as the ones included in this collection https://archive.org/details/georgeblood. It has shown to be robust to a wide range of different noises, but it may produce some artifacts if you try to inference in something completely different.

We used classical music as training data, so it is expected to work better with this genre than any other. Nevertheless, we also experienced good results with other kinds of non-classical music like, for instance, some old jazz recordings.

## Docker

Вы можете собрать и запустить этот проект с помощью Docker.

### Сборка образа

```bash
docker build -t historical-denoiser .
```

### Запуск обработки аудиофайла (inference)

Для обработки аудиофайла с помощью Docker используйте следующую команду. Контейнер будет автоматически удален после завершения (`--rm`) и будет иметь имя `war_denoise` во время работы.

```bash
docker run --rm --name war_denoise -v "/путь/к/вашей/папке_с_данными:/app/data" historical-denoiser python inference.py inference.input_file=/app/data/путь/к/входному/файлу.flac inference.output_dir=/app/data/путь/к/выходной/папке
```

**Разбор команды:**

*   `--rm`: Автоматически удалить контейнер после завершения.
*   `--name war_denoise`: Присвоить имя `war_denoise` работающему контейнеру.
*   `-v "/путь/к/вашей/папке_с_данными:/app/data"`:
    *   Монтирует вашу локальную папку (например, `E:\hacaton\docs\mus_for_denoise`) в папку `/app/data` внутри контейнера.
    *   Замените `/путь/к/вашей/папке_с_данными` на актуальный путь на вашем компьютере.
*   `historical-denoiser`: Имя Docker-образа.
*   `python inference.py`: Команда для запуска скрипта обработки.
*   `inference.input_file=/app/data/путь/к/входному/файлу.flac`:
    *   Параметр Hydra, указывающий путь к входному аудиофайлу *внутри контейнера*.
    *   Например, если ваша папка с данными смонтирована как `/app/data`, и входной файл лежит в `input/audio.flac` относительно этой папки, то путь будет `/app/data/input/audio.flac`.
*   `inference.output_dir=/app/data/путь/к/выходной/папке`:
    *   Параметр Hydra, указывающий путь к директории *внутри контейнера*, куда будут сохранены обработанные файлы.
    *   Например, `/app/data/output`.
    *   Скрипт создаст в этой директории три файла: `имя_файла_denoised.wav`, `имя_файла_noisy_input.wav`, и `имя_файла_residual.wav`.

**Пример для Windows:**

Если ваши файлы находятся в `E:\hacaton\docs\mus_for_denoise`, входной файл `input\9_war_denoise.flac`, и вы хотите сохранить результат в `output`: 

```bash
docker run --rm --name war_denoise -v "E:\hacaton\docs\mus_for_denoise:/app/data" historical-denoiser python inference.py inference.input_file=/app/data/input/9_war_denoise.flac inference.output_dir=/app/data/output
```

### Другие команды

Для запуска других команд, например, обучения (`train.py`) или интерактивной сессии `bash`:

Запуск обучения (пример, может потребовать настройки конфигурации Hydra для `train.py`):
```bash
docker run -v "/путь/к/вашей/папке_с_данными:/app/data" historical-denoiser python train.py <ваши_параметры_hydra_для_train>
```

Интерактивный режим:
```bash
docker run --rm -it -v "/путь/к/вашей/папке_с_данными:/app/data" historical-denoiser /bin/bash
```
Это запустит `bash` сессию внутри контейнера в активированном conda-окружении `historical_denoiser`.


