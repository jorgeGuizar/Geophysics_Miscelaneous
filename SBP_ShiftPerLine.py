import os
import re
import sys
import traceback
import collections
import shutil
from UTILS import *

file=r'D:\SEGY\SEGY_Files\OPEX\SBP_SEGY\MISTIE Table Report_240724_145628.settings'


file_out=r'D:\SEGY\SEGY_Files\OPEX\SBP_SEGY\shifts_file.txt'



read_shift_file(file,st1="SurveyName=",st2="TimeShift=",dest_file=file_out)

