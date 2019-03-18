#!/usr/bin/env python

import pandas as pd

usetypes={'Network' : 'str',
          'Station' : 'str',
          'Location' : 'str',
          'Channel' : 'str'}

# add error handling for 404 from post

df = pd.read_csv('pass_rrds.txt', skiprows=2, names=['Network', 'Station', 'Location', 'Channel', 'StartTime', 'EndTime'],dtype=usetypes, sep=' ')

df['Target'] = df[['Network', 'Station', 'Location', 'Channel']].apply(lambda row: '.'.join(row.values.astype(str))+'.M', axis=1)

df['SampleRate'] = 100

df.to_csv('pass_rrds.csv', sep='|', index=False)
