from music21 import *
import pickle

s = converter.parse('一首簡單的歌_主旋律.musicxml')
s.write('midi', fp='一首簡單的歌_主旋律.midi')