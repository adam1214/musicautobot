from musicautobot.numpy_encode import *
from musicautobot.utils.file_processing import process_all, process_file
from musicautobot.config import *
from musicautobot.music_transformer import *
from musicautobot.multitask_transformer import *
from musicautobot.numpy_encode import stream2npenc_parts
from musicautobot.utils.setup_musescore import setup_musescore
from music21 import *
import pickle

s = converter.parse('一首簡單的歌_主旋律.musicxml')
s.write('midi', fp='data/midi/my_example/一首簡單的歌_主旋律.mid')
midi_file = Path('data/midi/my_example/一首簡單的歌_主旋律.mid')

pred_melody = s2s_predict_from_midi(learn, midi_file, n_words=20, seed_len=4, pred_melody=True)