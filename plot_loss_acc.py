import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# first need to read in training output and grab the loss and acc values along with epoch and batch number

# open file
f = open("ResNet_train_gpu.sh.o1695330")
g = f.readlines()
f.close()

epoch = 0
batch = 0
loss = 0
acc = 0

results = []

for line in g:

    current_line = line
    if current_line != "":
        split_line = current_line.split()

        if len(split_line) > 0:
            if split_line[0] == "Epoch":
             epoch = split_line[1].replace('/75', '')

            elif len(split_line) == 11:
                batch = split_line[0].replace('/6573', '')
                loss = split_line[7]
                acc = split_line[10]
                if acc == "0.0000e+00":
                    acc = 0

    if epoch != 0 and batch != 0 and loss != 0:
        results.append([epoch, batch, loss, acc])


results = pd.DataFrame(results, columns=['Epoch', 'Batch', 'Loss', 'Accuracy'])
results= results.astype(float)
# get current axis
#results.plot(kind='line', y='Loss')
results.plot(kind='line', y='Accuracy')

plt.show()









