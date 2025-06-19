import os
basedir = r'C:\Users\LEGA\Documents\jorge\test'

for fn in os.listdir(basedir):
    print(fn)
    if not os.path.isdir(os.path.join(basedir, fn)):
        continue # Not a directory
    if '_' in fn:
        continue # Already in the correct form
    if ' ' not in fn:
        continue # Invalid format
    firstname,_,surname = fn.rpartition(' ')
    os.rename(os.path.join(basedir, fn),
            os.path.join(basedir, surname + '_' + firstname))