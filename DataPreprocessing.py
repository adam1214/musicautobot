from musicautobot.numpy_encode import *
from musicautobot.utils.file_processing import process_all, process_file
from musicautobot.config import *
from musicautobot.music_transformer import *
from musicautobot.multitask_transformer import *
from musicautobot.utils.stacked_dataloader import StackedDataBunch
from musicautobot.utils.midifile import *


from fastai.text import *

import random

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

def timeout_func(data, seconds):
    print("Timeout:", seconds)

def process_metadata(midi_file):
    # Get outfile and check if it exists
    out_file = s2s_numpy_path/midi_file.relative_to(midi_path).with_suffix('.npy')
    out_file.parent.mkdir(parents=True, exist_ok=True)
    if out_file.exists(): return
    
    npenc = transform_midi(midi_file)
    if npenc is not None: np.save(out_file, npenc)

def transform_midi(midi_file):
    input_path = midi_file
    
    try: 
        #if num_piano_tracks(input_path) not in [1, 2]: return None
        input_file = compress_midi_file(input_path, min_variation=min_variation, cutoff=2, supported_types=set([Track.PIANO])) # remove non note tracks and standardize instruments
        if not input_file: return None
    except Exception as e:
        if 'badly form' in str(e): return None # ignore badly formatted midi errors
        if 'out of range' in str(e): return None # ignore badly formatted midi errors
        print('Error parsing midi', input_path, e)
        return None
        
    # Part 2. Compress rests and long notes
    stream = file2stream(input_file) # 1.
    try:
        chordarr = stream2chordarr(stream) # 2. max_dur = quarter_len * sample_freq (4). 128 = 8 bars
    except Exception as e:
        print('Could not encode to chordarr:', input_path, e)
#         print(traceback.format_exc())
        return None
    
    chord_trim = trim_chordarr_rests(chordarr)
    chord_short = shorten_chordarr_rests(chord_trim)
    delta_trim = chord_trim.shape[0] - chord_short.shape[0]
#     if delta_trim > 300: 
#         print(f'Removed {delta_trim} rests from {input_path}. Skipping song')
#         return None
    chordarr = chord_short
    
    # Only 2 piano parts allowed
    if len(chordarr.shape) != 3: return None
    _,num_parts,_ = chordarr.shape
    print(num_parts)
    if num_parts != 2: return None
    
    # Individual parts must have notes
    parts = [part_enc(chordarr, i) for i in range(num_parts)]
    for p in parts: 
        if not is_valid_npenc(p, min_notes=8, input_path=input_path): return None
    
    # order by melody > chords
    p1, p2 = parts
    m, c = (p1, p2) if avg_pitch(p1) > avg_pitch(p2) else (p2, p1) # Assuming melody has higher pitch
    
    return np.array([m, c])

# Location of your midi files
midi_path = Path('data/midi/19_examples')
# Location of preprocessed numpy files
s2s_numpy_path = Path('data/numpy/multitask_preprocessed_data_s2s')

# Location of models and cached dataset
data_path = Path('data/cached')

lm_data_save_name = 'example_19_multitask_musicitem_data_save.pkl'
s2s_data_save_name = 'example_19_multitask_multiitem_data_save.pkl'

# num_tracks = [1, 2] # number of tracks to support
cutoff = 5 # max instruments
min_variation = 3 # minimum number of different midi notes played
# max_dur = 128

midi_files = get_files(midi_path, '.mid', recurse=True)


# # sanity check
for r in random.sample(midi_files, 10):
    process_metadata(r)

processed = process_all(process_metadata, midi_files, timeout=120, timeout_func=timeout_func)


processors = [Midi2ItemProcessor()]
data = MusicDataBunch.from_files(midi_files, data_path, processors=processors, 
                                 encode_position=True, dl_tfms=mask_lm_tfm_pitchdur, 
                                 bptt=5, bs=2)
data.save(lm_data_save_name)
