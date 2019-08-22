import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
# first need to read in training output and grab the loss and acc values along with epoch and batch number

# open file
file = "finished_runs/ResNet_train_gpu.sh.o1695330"
f = open(file)
g = f.readlines()
f.close()

run = file.split('.')[2]
outDir = "E:\\Documents\\CompSci\\project python files\\plots\\{}".format(run)
os.mkdir(outDir)


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
        val_results.append([val_loss, val_acc, epoch])
        val_loss = 0
        val_acc = 0


results = pd.DataFrame(results, columns=['Epoch', 'Batch', 'Loss', 'Accuracy'])
val_results = pd.DataFrame(val_results, columns=['val_loss', 'val_acc', 'Epoch'])
val_results = val_results.astype(float)
results = results.astype(float)
# get current axis
plt.plot(results.Loss)
plt.title('Training Loss')
plt.xlabel('Iterations of training')
plt.ylabel('Measured Loss')
plt.savefig('{}\\{}'.format(outDir, 'train_loss'))
plt.show()


plt.plot(results.Accuracy)
plt.title('Training Accuracy')
plt.xlabel('Iterations of training')
plt.ylabel('Measured Accuracy')
plt.savefig('{}\\{}'.format(outDir, 'train_acc'))
plt.show()

plt.scatter(y=val_results['val_loss'], x=val_results['Epoch'])
plt.title('Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Measured Loss')
plt.savefig('{}\\{}'.format(outDir, 'val_loss'))
plt.show()

plt.scatter(y=val_results['val_acc'], x=val_results['Epoch'])
plt.title('Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Measured Loss')
plt.savefig('{}\\{}'.format(outDir, 'val_acc'))
plt.show()

#results.plot(kind='line', y='Accuracy')











