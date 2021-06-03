from musicautobot.numpy_encode import *
from musicautobot.utils.file_processing import process_all, process_file
from musicautobot.config import *
from musicautobot.music_transformer import *
from musicautobot.multitask_transformer import *
from musicautobot.utils.stacked_dataloader import StackedDataBunch

from fastai.text import *

# Location of your midi files
midi_path = Path('data/midi/19_examples')
midi_path.mkdir(parents=True, exist_ok=True)

# Location to save dataset
data_path = Path('data/numpy')
data_path.mkdir(parents=True, exist_ok=True)

data_save_name = 'musicitem_data_save.pkl'
s2s_data_save_name = 'multiitem_data_save.pkl'

midi_files = get_files(midi_path, '.mid', recurse=True)

# Create NextWord/Mask Dataset
print('Create NextWord/Mask Dataset')
processors = [Midi2ItemProcessor()]
data = MusicDataBunch.from_files(midi_files, data_path, processors=processors, 
                                 encode_position=True, dl_tfms=mask_lm_tfm_pitchdur, 
                                 bptt=5, bs=2)
data.save(data_save_name)
xb, yb = data.one_batch()

# Create sequence to sequence dataset
print('Create sequence to sequence dataset')
processors = [Midi2MultitrackProcessor()]
s2s_data = MusicDataBunch.from_files(midi_files, data_path, processors=processors, 
                                     preloader_cls=S2SPreloader, list_cls=S2SItemList,
                                     dl_tfms=melody_chord_tfm,
                                     bptt=5, bs=2)
s2s_data.save(s2s_data_save_name)
xb, yb = s2s_data.one_batch()

# Initialize Model
print('Initialize Model')
# Load Data
print('Load Data')
batch_size = 2
bptt = 128

lm_data = load_data(data_path, data_save_name, 
                    bs=batch_size, bptt=bptt, encode_position=True,
                    dl_tfms=mask_lm_tfm_pitchdur)

s2s_data = load_data(data_path, s2s_data_save_name, 
                     bs=batch_size//2, bptt=bptt,
                     preloader_cls=S2SPreloader, dl_tfms=melody_chord_tfm)

# Combine both dataloaders so we can train multiple tasks at the same time
print('Combine both dataloaders so we can train multiple tasks at the same time')
data = StackedDataBunch([lm_data, s2s_data])

# Create Model
print('Create Model')
config = multitask_config()

learn = multitask_model_learner(data, config.copy())
# learn.to_fp16(dynamic=True) # Enable for mixed precision

print('Training...')
learn.fit_one_cycle(4)

print('Saving the model...')
learn.save('my_model')