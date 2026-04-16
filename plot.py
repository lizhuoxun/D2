import numpy as np
import argparse
import matplotlib.pyplot as plt

# 设置全局字体为 Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 30 

parser = argparse.ArgumentParser()
parser.add_argument('--traingen', default='facevid2vid', help='generator of training')
parser.add_argument('--savedir', default='./result')

opt = parser.parse_args()
print(opt)

test_gen = ["facevid2vid", "lia", "tps"]
line_styles = ['-', '--', ':']  
colors = ['blue', 'green', 'red'] 

plt.figure(figsize=(10, 8))

for idx, gen in enumerate(test_gen):
    with np.load(f"{opt.savedir}/{opt.traingen}/fpr_{gen}.npz") as data:
        fpr = data['macro']
    with np.load(f"{opt.savedir}/{opt.traingen}/tpr_{gen}.npz") as data:
        tpr = data['macro']
    with np.load(f"{opt.savedir}/{opt.traingen}/auc_score_{gen}.npz") as data:
        auc = data['macro']


    plt.plot(fpr, tpr,
             linestyle=line_styles[idx],
             color=colors[idx],
             linewidth=2,
             label=f"{gen.upper()} - {auc:.2f}")

plt.xticks(fontsize=20)  
plt.yticks(fontsize=20)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate', fontsize=30)
plt.ylabel('True Positive Rate', fontsize=30)
plt.title('Trained on %s'%opt.traingen.upper(), fontsize=30)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='lower right', fontsize=30)
plt.tight_layout()

plt.savefig('%s/%s/combined_roc_curves.pdf' % (opt.savedir, opt.traingen), dpi=300, bbox_inches='tight')
plt.show()
