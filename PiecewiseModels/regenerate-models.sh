#!/bin/bash

python resample2peterson.py ../peterson_HNM.mod High_T-vs-dB.txt wolinmcnamara_HNM.mod

python resample2peterson.py ../peterson_LNM.mod Low_Perm_T-vs-dB.txt wolinmcnamara_Perm_LNM.mod

python resample2peterson.py ../peterson_LNM.mod Low_Port_T-vs-dB.txt wolinmcnamara_Port_LNM.mod
