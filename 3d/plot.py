import matplotlib.pyplot as plt
import csv
import glob


BASE_DIR = './plot/'
list_name = ['0.6/','0.7/','0.8/']
list_plot = ['run-train-tag-epoch_loss.csv', 'run-validation-tag-epoch_loss.csv',
            'run-train-tag-epoch_Precision.csv', 'run-validation-tag-epoch_Precision.csv',
            'run-train-tag-epoch_Recall.csv','run-validation-tag-epoch_Recall.csv']

def plot(fname, csvf, label, xlable, ylable):
    x = []
    y = [] 
    csvfile = open(BASE_DIR+fname+csvf,'r')
    plots = csv.reader(csvfile)
    next(plots)
    for row in plots:
        x.append(float(row[1]))
        y.append(float(row[2]))
    plt.plot(x,y, label=label)
    plt.xlabel(xlable)
    plt.ylabel(ylable)

for model in list_name:
    plot(model, list_plot[0], 'Training', 'epoch', 'loss')
    plot(model, list_plot[1], 'Validation', 'epoch', 'loss')
    plt.title('Loss')
    plt.legend()
    plt.savefig(BASE_DIR+model+'Loss.png')
    plt.close()

    plot(model, list_plot[2], 'Training', 'epoch', 'precision')
    plot(model, list_plot[3], 'Validation', 'epoch', 'precision')
    plt.title('Precision')
    plt.legend()
    plt.savefig(BASE_DIR+model+'Precision.png')
    plt.close()

    plot(model, list_plot[4], 'Training', 'epoch', 'recall')
    plot(model, list_plot[5], 'Validation', 'epoch', 'recall')
    plt.title('Recall')
    plt.legend()
    plt.savefig(BASE_DIR+model+'Recall.png')
    plt.close()