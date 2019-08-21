import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# first need to read in training output and grab the loss and acc values along with epoch and batch number

# open file
f = open("finished_runs/ResNet_train_gpu.sh.o1695486")
g = f.readlines()
f.close()

epoch = 0
batch = 0
loss = 0
acc = 0
val_loss = 0
val_acc = 0

results = []
val_results = []

for line in g:

    current_line = line
    if current_line != "":
        split_line = current_line.split()

        if len(split_line) > 0:
            if split_line[0] == "Epoch":
             epoch = split_line[1].split('/')[0]

            elif len(split_line) == 11:

                batch = split_line[0].split('/')[1]
                loss = split_line[7]
                acc = split_line[10]
                if acc == "0.0000e+00":
                    acc = 0
            elif len(split_line) == 17:
                val_loss = split_line[13]
                val_acc = split_line[16]

    if epoch != 0 and batch != 0 and loss != 0:
        results.append([epoch, batch, loss, acc])

    if val_loss != 0 and val_acc != 0:
        val_results.append([val_loss, val_acc])
        val_loss = 0
        val_acc = 0


results = pd.DataFrame(results, columns=['Epoch', 'Batch', 'Loss', 'Accuracy'])
val_results = pd.DataFrame(val_results, columns=['val_loss', 'val_acc'])
val_results = val_results.astype(float)
results = results.astype(float)
# get current axis
results.plot(kind='line', y='Loss')
plt.show()
results.plot(kind='line', y='Accuracy')
plt.show()

val_results.plot(y='val_loss')
plt.show()
val_results.plot(y='val_acc')
plt.show()

#results.plot(kind='line', y='Accuracy')











