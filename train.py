from musicautobot.numpy_encode import *
from musicautobot.utils.file_processing import process_all, process_file
from musicautobot.config import *
from musicautobot.music_transformer import *
from musicautobot.multitask_transformer import *
from musicautobot.utils.stacked_dataloader import StackedDataBunch
from musicautobot.utils.midifile import *


from fastai.text import *

def create_databunch(files, data_save_name, path):
    save_file = path/data_save_name
    if save_file.exists():
        data = load_data(path, data_save_name)
    else:
        save_file.parent.mkdir(exist_ok=True, parents=True)
        vocab = MusicVocab.create()
        processors = [S2SFileProcessor(), S2SPartsProcessor()]

        data = MusicDataBunch.from_files(files, path, processors=processors, 
                                          preloader_cls=S2SPreloader, list_cls=S2SItemList, dl_tfms=melody_chord_tfm)
        data.save(data_save_name)
    return data

# Location of your midi files
midi_path = Path('data/midi/19_examples')
# Location of preprocessed numpy files
s2s_numpy_path = Path('data/numpy/multitask_preprocessed_data_s2s')

# Location of models and cached dataset
data_path = Path('data/cached')

lm_data_save_name = 'example_19_multitask_musicitem_data_save.pkl'
s2s_data_save_name = 'example_19_multitask_multiitem_data_save.pkl'

print('Create sequence to sequence dataset')
s2s_numpy_files = get_files(s2s_numpy_path, extensions='.npy', recurse=True)
#s2s_data = create_databunch(s2s_numpy_files, data_save_name=s2s_data_save_name, path=data_path)
#xb, yb = s2s_data.one_batch()

print('Initialize Model')
# Load Data
batch_size = 2
bptt = 128

lm_data = load_data(data_path, lm_data_save_name, 
                    bs=batch_size, bptt=bptt, encode_position=True,
                    dl_tfms=mask_lm_tfm_pitchdur)

s2s_data = load_data(data_path, s2s_data_save_name, 
                     bs=batch_size//2, bptt=bptt,
                     preloader_cls=S2SPreloader, dl_tfms=melody_chord_tfm)

# Combine both dataloaders so we can train multiple tasks at the same time
data = StackedDataBunch([lm_data, s2s_data])

# Create Model
config = multitask_config()

learn = multitask_model_learner(data, config.copy())
# learn.to_fp16(dynamic=True) # Enable for mixed precision

learn.fit_one_cycle(4)
learn.save('examples_19_multitask_model')