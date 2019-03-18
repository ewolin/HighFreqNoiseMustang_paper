#!/usr/bin/env python

import pandas as pd

from addIRISPDFs import readIRISfedcat

df = readIRISfedcat('irisfedcat_samprate_ge100.txt')

# replace empty net codes with --
mask = df['Location'] == ''
df.loc[mask, 'Location'] = '--'

header = ['Network', 'Station', 'Location', 'Channel', 'StartTime', 'EndTime']
df.to_csv('df2post.csv', columns=header, sep=' ', index=False)


