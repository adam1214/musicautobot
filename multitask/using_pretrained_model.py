from musicautobot.numpy_encode import *
from musicautobot.utils.file_processing import process_all, process_file
from musicautobot.config import *
from musicautobot.music_transformer import *
from musicautobot.multitask_transformer import *
from musicautobot.numpy_encode import stream2npenc_parts
from musicautobot.utils.setup_musescore import setup_musescore
from music21 import *
import pickle
setup_musescore()

from midi2audio import FluidSynth
from IPython.display import Audio

# Colab cannot play music directly from music21 - must convert to .wav first
def play_wav(stream):
    out_midi = stream.write('midi')
    out_wav = str(Path(out_midi).with_suffix('.wav'))
    FluidSynth("font.sf2").midi_to_audio(out_midi, out_wav)
    return Audio(out_wav)

s = converter.parse('一首簡單的歌_主旋律.musicxml')
s.write('midi', fp='data/midi/my_example/一首簡單的歌_主旋律.mid')

# Config
# Load Pretrained
config = multitask_config()

# Location of your midi files
midi_path =  Path('data/midi')

# Location of saved datset
data_path = Path('data/numpy')
data_save_name = 'musicitem_data_save.pkl'

# Data
data = MusicDataBunch.empty(data_path)
vocab = data.vocab
'''
# Pretrained Model

# Download pretrained model if you haven't already
# pretrained_url = 'https://ashaw-midi-web-server.s3-us-west-2.amazonaws.com/pretrained/MultitaskSmallKeyC.pth'
pretrained_url = 'https://ashaw-midi-web-server.s3-us-west-2.amazonaws.com/pretrained/MultitaskSmall.pth'

pretrained_path = data_path/'pretrained'/Path(pretrained_url).name
pretrained_path.parent.mkdir(parents=True, exist_ok=True)
download_url(pretrained_url, dest=pretrained_path)
'''
pretrained_path = data_path/'pretrained/my_model.pth'
# Learner
learn = multitask_model_learner(data, pretrained_path=pretrained_path, config=multitask_config().copy()) # 自己train的model用這行
# learn = multitask_model_learner(data, pretrained_path=pretrained_path) # 原作的model用這行

example_dir = midi_path/'my_example'
midi_files = get_files(example_dir, recurse=True, extensions='.mid')

file = midi_files[0]

# Encode file 
item = MusicItem.from_file(file, data.vocab)

x = item.to_tensor()
x_pos = item.get_pos_tensor()

#item.show()

# item.play()
#play_wav(item.stream)

multitrack_item = MultitrackItem.from_file(file, vocab)
melody, chords = multitrack_item.melody, multitrack_item.chords
#melody.show()
#chords.show()
#multitrack_item.play()
#play_wav(multitrack_item.stream)

#Generate chords to accompany an existing melody
# partial_chords = chords.trim_to_beat(3);
# partial_chords.show()

empty_chords = MusicItem.empty(vocab, seq_type=SEQType.Chords)

pred_chord = learn.predict_s2s(input_item=melody, target_item=empty_chords)
#pred_chord.show()

combined = MultitrackItem(melody, pred_chord)
combined.play()

combined.stream.write('midi',fp='Accompaniment.mid')
#combined.show()

#play_wav(combined.stream)