# Want to extract only the images within folder 0 to leave a usable sized batch

import shutil
import os

source = 'E:\\Documents\\CompSci\\project\\trainset'
destination = 'E:\\Documents\\CompSci\\project\\trainset\\'

subfolders1 = os.listdir(source)

# Remove conditionals to extract all images

for f in subfolders1:
    subfolders2 = os.listdir(source + '\\' + f)
    for f2 in subfolders2:
        subfolders3 = os.listdir(source + '\\' + f + '\\' + f2)
        for f3 in subfolders3:
            subfolders4 = os.listdir(source + '\\' + f + '\\' + f2 + '\\' + f3)
            for f4 in subfolders4:
                # if f == '0':
                # files = os.listdir(source + '\\' + f + '\\' + f2 + '\\' + f3 + '\\' + f4)
                # for f_lowest in files:
                file = source + '\\' + f + '\\' + f2 + '\\' + f3 + '\\' + f4
                print(file)
                shutil.move(file, destination)
