# @pierrotechnique
# -*- coding: utf-8 -*-

import pyo
#import librosa as lr

s = pyo.Server(nchnls=1)

s.boot()

s.start()

f1 = 440.
a = pyo.Sine(f1).out()

# Initialize table to record final signal samples
t = pyo.NewTable(0.02,1)
# NewTable(len_sec,nchnls)

# Record signal
rec = pyo.TrigTableRec(a,trig=pyo.Trig().play(),table=t)
# TableRec(signal,table,fadeIn_sec)

s.stop()

x = rec.get(440)