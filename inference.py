import os
import hydra
import logging
import glob # Для поиска файлов

logger = logging.getLogger(__name__)

# Определим поддерживаемые расширения аудиофайлов
SUPPORTED_EXTENSIONS = ('.wav', '.flac', '.mp3', '.ogg', '.m4a')

def process_single_file(input_audio_path, output_audio_dir, args, unet_model):
    """Обрабатывает один аудиофайл."""
    import tensorflow as tf # Импорты здесь, чтобы не загружать TF, если нет файлов
    import soundfile as sf
    import numpy as np
    import scipy.signal
    from tqdm import tqdm # Можно убрать tqdm для пакетной обработки или оставить

    logger.info(f"Обработка файла: {input_audio_path}")
    os.makedirs(output_audio_dir, exist_ok=True)

    def do_stft(noisy):
        window_fn = tf.signal.hamming_window
        win_size=args.stft.win_size
        hop_size=args.stft.hop_size
        stft_signal_noisy=tf.signal.stft(noisy,frame_length=win_size, window_fn=window_fn, frame_step=hop_size, pad_end=True)
        stft_noisy_stacked=tf.stack( values=[tf.math.real(stft_signal_noisy), tf.math.imag(stft_signal_noisy)], axis=-1)
        return stft_noisy_stacked

    def do_istft(data):
        window_fn = tf.signal.hamming_window
        win_size=args.stft.win_size
        hop_size=args.stft.hop_size
        inv_window_fn=tf.signal.inverse_stft_window_fn(hop_size, forward_window_fn=window_fn)
        pred_cpx=data[...,0] + 1j * data[...,1]
        pred_time=tf.signal.inverse_stft(pred_cpx, win_size, hop_size, window_fn=inv_window_fn)
        return pred_time
    try:
        data, samplerate = sf.read(input_audio_path)
    except Exception as e:
        logger.error(f"Не удалось прочитать входной файл {input_audio_path}: {e}")
        return

    #Стерео в моно
    if len(data.shape)>1:
        data=np.mean(data,axis=1)
    
    if samplerate!=44100: 
        logger.info(f"Изменение частоты дискретизации файла {input_audio_path} с {samplerate} Гц на 44100 Гц")
        data=scipy.signal.resample(data, int((44100  / samplerate )*len(data))+1)  
 
    segment_size=44100*5  #5s segments
    length_data=len(data)
    overlapsize=2048 #samples (46 ms)
    window=np.hanning(2*overlapsize)
    window_right=window[overlapsize::]
    window_left=window[0:overlapsize]
    
    denoised_data=np.zeros(shape=(len(data),))
    residual_noise_data=np.zeros(shape=(len(data),))
    numchunks=int(np.ceil(length_data/segment_size))
    pointer = 0
     
    for i in tqdm(range(numchunks), desc=f"Подавление шума в {os.path.basename(input_audio_path)}"):
        current_segment_data = data[pointer : pointer + segment_size]
        is_last_chunk = (pointer + segment_size >= length_data)

        if is_last_chunk:
            actual_segment_length = len(current_segment_data)
            padded_segment_data = np.pad(current_segment_data, (0, segment_size - actual_segment_length), 'constant')
        else:
            padded_segment_data = current_segment_data
            actual_segment_length = segment_size

        segment_TF = do_stft(padded_segment_data)
        segment_TF_ds = tf.data.Dataset.from_tensors(segment_TF)
        pred_batch = unet_model.predict(segment_TF_ds.batch(1))
        pred_TF = pred_batch[0] # (1, H, W, C) -> (H, W, C)
        
        residual_TF = segment_TF - pred_TF
        
        pred_time_full = do_istft(pred_TF)
        residual_time_full = do_istft(residual_TF)

        # Обрезаем до актуальной длины сегмента (важно для последнего чанка)
        pred_time = pred_time_full[:actual_segment_length]
        residual_time = residual_time_full[:actual_segment_length]

        # Применяем windowing (overlap-add)
        if numchunks == 1: # Если всего один чанк, windowing не нужен
            processed_pred_time = pred_time
            processed_residual_time = residual_time
        elif pointer == 0: # Первый чанк
            processed_pred_time = np.concatenate((pred_time[:actual_segment_length-overlapsize], pred_time[actual_segment_length-overlapsize:actual_segment_length] * window_right[:len(pred_time[actual_segment_length-overlapsize:actual_segment_length])]), axis=0)
            processed_residual_time = np.concatenate((residual_time[:actual_segment_length-overlapsize], residual_time[actual_segment_length-overlapsize:actual_segment_length] * window_right[:len(residual_time[actual_segment_length-overlapsize:actual_segment_length])]), axis=0)
        elif is_last_chunk: # Последний чанк (но не первый)
            processed_pred_time = np.concatenate((pred_time[:overlapsize] * window_left, pred_time[overlapsize:]), axis=0)
            processed_residual_time = np.concatenate((residual_time[:overlapsize] * window_left, residual_time[overlapsize:]), axis=0)
        else: # Средний чанк
            processed_pred_time = np.concatenate((pred_time[:overlapsize] * window_left, pred_time[overlapsize:actual_segment_length-overlapsize], pred_time[actual_segment_length-overlapsize:actual_segment_length] * window_right[:len(pred_time[actual_segment_length-overlapsize:actual_segment_length])]), axis=0)
            processed_residual_time = np.concatenate((residual_time[:overlapsize] * window_left, residual_time[overlapsize:actual_segment_length-overlapsize], residual_time[actual_segment_length-overlapsize:actual_segment_length] * window_right[:len(residual_time[actual_segment_length-overlapsize:actual_segment_length])]), axis=0)
        
        # Собираем результат
        current_output_slice = denoised_data[pointer : pointer + actual_segment_length]
        current_output_slice_len = len(current_output_slice)
        denoised_data[pointer : pointer + actual_segment_length] += processed_pred_time[:current_output_slice_len]
        
        current_residual_slice = residual_noise_data[pointer : pointer + actual_segment_length]
        current_residual_slice_len = len(current_residual_slice)
        residual_noise_data[pointer : pointer + actual_segment_length] += processed_residual_time[:current_residual_slice_len]

        if not is_last_chunk:
            pointer += segment_size - overlapsize
        else:
            pointer += actual_segment_length # Переход к концу файла

    # Сохраняем оригинальный  вход
    noisy_input_path = os.path.join(output_audio_dir, "noisy_input.wav")
    sf.write(noisy_input_path, data, 44100)
    logger.info(f"Сохранен исходный входной файл (моно, 44.1кГц): {noisy_input_path}")

    denoised_output_path = os.path.join(output_audio_dir, "denoised.wav")
    sf.write(denoised_output_path, denoised_data, 44100)
    logger.info(f"Сохранен обработанный файл: {denoised_output_path}")

    residual_output_path = os.path.join(output_audio_dir, "residual.wav")
    sf.write(residual_output_path, residual_noise_data, 44100)
    logger.info(f"Сохранен остаточный шум: {residual_output_path}")

def run(args):
    import unet
    
    CONTAINER_INPUT_DIR = "/app/data/input/"
    CONTAINER_OUTPUT_DIR = "/app/data/output/"

    path_experiment=str(args.path_experiment)
    ckpt_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),path_experiment, 'checkpoint')

    if not os.path.exists(ckpt_path + '.index'):
         logger.error(f"Файл контрольной точки не найден по адресу {ckpt_path}. Убедитесь, что модели загружены и распакованы правильно.")
         os._exit(1)

    unet_model = unet.build_model_denoise(unet_args=args.unet)
    unet_model.load_weights(ckpt_path)
    logger.info(f"Модель загружена из {ckpt_path}")

    input_files = []
    for ext in SUPPORTED_EXTENSIONS:
        input_files.extend(glob.glob(os.path.join(CONTAINER_INPUT_DIR, f"*{ext}")))
        input_files.extend(glob.glob(os.path.join(CONTAINER_INPUT_DIR, f"*{ext.upper()}")))

    if not input_files:
        logger.warning(f"Аудиофайлы не найдены в директории {CONTAINER_INPUT_DIR}")
        return

    logger.info(f"Найдено {len(input_files)} файлов для обработки: {input_files}")

    for input_file_path in input_files:
        filename_stem = os.path.splitext(os.path.basename(input_file_path))[0]
        output_file_specific_dir = os.path.join(CONTAINER_OUTPUT_DIR, filename_stem)
        
        try:
            process_single_file(input_file_path, output_file_specific_dir, args, unet_model)
        except Exception as e:
            logger.error(f"Ошибка при обработке файла {input_file_path}: {e}", exc_info=True)

def _main(args):
    global __file__
    __file__ = hydra.utils.to_absolute_path(__file__)
    run(args)

@hydra.main(config_path="conf/conf.yaml")
def main(args):
    try:
        _main(args)
    except Exception:
        logger.exception("Произошла ошибка")
        os._exit(1)

if __name__ == "__main__":
    main()







