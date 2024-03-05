from os import makedirs


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.colors as mpl
import pandas as pd

import numpy as np


class SyntheticData:

    def __init__(self, seed, root_directory=None):
        self.seed = seed
        self.root_directory = root_directory

    @staticmethod
    def get_synthetic_features_and_classes(with_redundant_column=False):
        features = [
            '(1) A',
            '(2) B',
        ]
        if with_redundant_column:
            features.append('(3) R')

        # classes = ['EVEN', 'ODD']
        classes = ['RED', 'BLUE']
        return features, classes

    def get_synthetic_data(self, x_dim=20, y_dim=20, add_noise=0.0, add_redundant_column=False):
        sd = self.__create_synthetic_data(x_dim, y_dim, add_noise, add_redundant_column)
        sd2 = sd.values

        output = None
        if self.root_directory is not None:
            output = self.root_directory + f'/synthetic__sz_{x_dim*y_dim}__n_{1 if add_noise else 0}__rc_{add_redundant_column}/'.lower().replace('.', '_')
            try:
                makedirs(output)
            except OSError as e:
                pass
        features, classes = self.get_synthetic_features_and_classes(add_redundant_column)
        return sd2, features, classes, output

    def setup_synthetic_data_test_train(self, data, test_size=0.30):
        x = np.array(data[:, 0:-1])
        y = np.array(data[:, -1])

        x_tr, x_ts, y_tr, y_ts = train_test_split(x, y,
                                                  test_size=test_size,
                                                  random_state=self.seed,
                                                  stratify=y)

        # Normalize
        s = MinMaxScaler()
        x_tr = s.fit_transform(x_tr)
        x_ts = s.transform(x_ts)
        x = s.transform(x)

        return x, y, x_tr, x_ts, y_tr, y_ts

    def __create_synthetic_data(self, x_dim, y_dim, add_noise=0.0, add_redundant_column=False):
        np.random.seed(self.seed)
        # clean data
        data = []

        x_mid = x_dim / 2
        y_mid = y_dim / 2
        xmrl = int(1 + x_mid + x_mid / 4)
        xmrh = int(xmrl + x_mid / 2)
        ymrl = int(1 + y_mid + y_mid / 4)
        ymrh = int(ymrl + y_mid / 2)

        xmll = int(x_mid / 4)
        xmlh = xmll + int(x_mid / 2)
        ymll = int(x_mid / 4)
        ymlh = ymll + int(y_mid / 2)

        for x in range(0, x_dim):
            for y in range(0, y_dim):
                # data.append([x, y, (x + y) % 2])
                value = 0 if (x + y) < (x_dim + y_dim) / 2 else 1
                r = np.random.random(1)[0]
                if (xmrl < x < xmrh and ymrl < y < ymrh) or (xmll < x < xmlh and ymll < y < ymlh):
                    data.append([x, y, r, 1 - value])
                else:
                    data.append([x, y, r, value])

        if add_noise > 0:
            noise_count = int((add_noise * x_dim * y_dim) + 0.5)
            # flip some point values
            for i in range(0, noise_count):
                x = np.random.randint(x_dim)
                y = np.random.randint(y_dim)
                xy = y + y_dim * x
                abrc = data[xy]
                r = np.random.random(1)[0]
                data[xy] = [x, y, r, 1-abrc[-1]]

            # duplicate some rows and randomly flip the data for those rows
            for i in range(0, noise_count * 2):
                # duplicate a random row and flip the data
                x = np.random.randint(x_dim)
                y = np.random.randint(y_dim)
                r = np.random.random(1)[0]
                xy = y + y_dim * x
                abrc = data[xy]
                vo = abrc[-1]
                ro = abrc[-2]
                data.append([x, y, ro, vo if r < 0.5 else 1-vo])

        df = pd.DataFrame.from_records(data)
        df.rename(columns={0: "A", 1: "B", 2: "R", 3: "C"}, errors="raise", inplace=True)
        if not add_redundant_column:
            df.drop(columns=['R'], inplace=True)
        return df


def plot_synthetic_dataset(x_train, x_test, y_train, y_test, classifier=None, transparent_bg=False, bg_color='white'):
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].

    o = 0.05
    x_min_train, x_max_train = x_train[:, 0].min() - o, x_train[:, 0].max() + o
    y_min_train, y_max_train = x_train[:, 1].min() - o, x_train[:, 1].max() + o
    x_min_test, x_max_test = x_test[:, 0].min() - o, x_test[:, 0].max() + o
    y_min_test, y_max_test = x_test[:, 1].min() - o, x_test[:, 1].max() + o
    x_min = min(x_min_test, x_min_train)
    y_min = max(y_min_test, y_min_train)
    x_max = min(x_max_test, x_max_train)
    y_max = max(y_max_test, y_max_train)

    has_3_columns = x_train.shape[1] == 3

    h = .02
    rr = None
    if has_3_columns:
        xx, yy, rr = np.meshgrid(np.arange(x_min, x_max, h),
                                 np.arange(y_min, y_max, h),
                                 np.arange(0, 1, h))
    else:
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

    # just plot the dataset first
    cm = plt.get_cmap('RdBu')
    cm_bright = mpl.ListedColormap(['#FF0000', '#0000FF'])
    # ax = plt.subplot(len(datasets), len(classifiers) + 1, i)

    ax = plt.gca()
    if classifier is not None:
        dd = np.c_[xx.ravel(), yy.ravel()] if not has_3_columns else np.c_[xx.ravel(), yy.ravel(), rr.ravel()]
        if hasattr(classifier, "decision_function"):
            Z = classifier.decision_function(dd)
        else:
            Z = classifier.predict_proba(dd)[:, 1]

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        if has_3_columns:
            xx = xx[:, :, 0]
            yy = yy[:, :, 0]
            Z = Z.mean(axis=2)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

    # Plot also the training points
    ax.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=cm_bright, edgecolor='darkgreen')
    # and testing points
    ax.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap=cm_bright, edgecolor='white', alpha=0.6)
    ax.patch.set_facecolor(bg_color)
    if transparent_bg:
        ax.patch.set_alpha(0)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    # ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
    #        size=15, horizontalalignment='right')

    plt.show()
