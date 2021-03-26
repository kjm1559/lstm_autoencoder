import numpy as np
import matplotlib.pyplot as plt

def cut_func(data):
    th = -0.1
    cut_index = []
    cut_data = []
    cut_index = np.sort(list(set(np.where(np.array(data) <= th)[0])))
    if len(cut_index) == 0:
        return np.expand_dims(data, 0)
    cut_data.append(data[:cut_index[0]])
    for i in range(len(cut_index) - 1):
        cut_data.append(data[cut_index[i]+1:cut_index[i + 1]])
    cut_data.append(data[cut_index[-1]:])
    return cut_data

def result_plot(index, model, data, label):
    target = np.expand_dims(np.array(data[index]), 0).astype('float')
    char = label[index]
    
    # prediction
    predict = model.predict(target)[:, ::-1, :]
    plt.subplot(1, 2, 1)
    tmp_target = cut_func(target[0])
    for tmp in tmp_target:
        plt.plot(tmp[:, 0], 1-tmp[:,1])
    plt.ylim(-0.5, 1.5)
    plt.xlim(-0.5, 1.5)
    plt.title(char)
    plt.subplot(1, 2, 2)
    tmp_predict = cut_func(predict[0])
    for tmp in tmp_predict:
        plt.plot(tmp[:, 0], 1-tmp[:,1])
    plt.ylim(-0.5, 1.5)
    plt.xlim(-0.5, 1.5)
    plt.title('generated')
    plt.show()

def penchar_parser(file='ujipenchars2.txt'):
    f = open(file, 'r')
    label = []
    data = []
    while True:
        line = f.readline()
        if not line: break
        if 'WORD' in line:
            label.append(line.split(' ')[1])
            line = f.readline()
            tmp_points = []
            pen_up = []
            for i in range(int(line.split(' ')[-1])):
                line = f.readline()
                points = line.split(' ')[3]
                points_str = line.split('#')[1]
                points_float = [float(ss) for ss in points_str[1:].split(' ')]
                points_float = np.array(points_float).reshape((int(points), 2))
                tmp_points += points_float.tolist()
                tmp_points += [tmp_points[-1]]
                pen_up.append(len(tmp_points) - 1)
            tmp_points = np.array(tmp_points)
            tmp_points = (tmp_points - np.min(tmp_points, axis=0))/(np.max(tmp_points, axis=0) - np.min(tmp_points, axis=0) + 1e-10)
            for i in pen_up:
                tmp_points[i] = [-1, -1]
            data.append(tmp_points.tolist())
    return label, data

class data_augmentation():
    def __init__(self, data, lambda_=0.5):
        '''
        initial function, set data
        Args:
            data (numpy.array): target data
            lambda_ (float): ratio
        '''
        self.target_data = data
        self.lambda_ = lambda_
        
    def interpolation_augmentation(self, data):
        '''
        interpolation function
        Args:
            data (numpy.array): interpolation target
        Returns:
            tmp_context (list): interpolation result
        '''
        k_n = self.clf.kneighbors(np.expand_dims(data, 0))
        tmp_context = []
        for ii in k_n[1][0][1:]:
            tmp_context.append((self.target_data[ii] - data) * self.lambda_ + data)
        return tmp_context
    
    def extrapolation_augmentation(self, data):
        '''
        extrapolation function
        Args:
            data (numpy.array): extrapolation target
        Returns:
            tmp_context (list): extrapolation result
        '''
        k_n = self.clf.kneighbors(np.expand_dims(data, 0))
        tmp_context = []
        for ii in k_n[1][0][1:]:
            tmp_context.append((self.target_data[ii] - data) * self.lambda_ + data)
        return tmp_context
    
    def hard_extrapolation_augmentation(self, data):
        '''
        hard extrapolation function
        Args:
            data (numpy.array): hard extrapolation target
        Returns:
            tmp_context (list): hard extrapolation result
        '''
        return np.expand_dims(data + self.lambda_ * (data + self.center), 0)
    
    def gaussian_noise(self, data, k):
        '''
        gausian noise function
        Args:
            data (numpy.array): gausian noise target
        Returns:
            tmp_context (list): gausian noise result
        '''
        tmp_context = (np.repeat(np.expand_dims(np.array(data), axis=0), k, axis=0) + self.lambda_ * np.random.normal(np.zeros(self.std.shape), self.std, (k, self.std.shape[0]))).tolist()
        return tmp_context
    
    def augmentation_init(self, da_type, k):
        '''
        '''
        self.da_type = da_type
        self.k = k
        if da_type == 'norm':
            self.std = np.std(self.target_data, axis=0)
        elif da_type == 'hard_extra':
            self.center = np.mean(self.target_data, axis=0)
        else:
            self.clf = NearestNeighbors(n_neighbors=k+1) # include self selection 
            self.clf.fit(self.target_data)
        
    
    def augmentation_process(self, da_type='norm', k=10):
        '''
        data augmentation process function
        Args:
            da_type (str): data augmentation type['norm', 'inter', 'extra', 'hard_extra']
        Returns:
            data (numpy.array): augmentation result data
            label (numpy.array): augmentation label
            class_weight (dict): class weight
        '''
        self.augmentation_init(da_type, k)
        tmp_context = []
        for j in range(self.target_data.shape[0]):
            if self.da_type=='norm':
                tmp_context += self.gaussian_noise(self.target_data[j], self.k)
            elif self.da_type == 'inter':
                tmp_context += self.interpolation_augmentation(self.target_data[j])
            elif self.da_type == 'extra':
                tmp_context += self.extrapolation_augmentation(self.target_data[j])
            elif self.da_type == 'hard_extra':
                tmp_context += self.hard_extrapolation_augmentation(self.target_data[j]).tolist()
        
        data = np.concatenate([self.target_data, np.array(tmp_context)], axis=0)
        label = np.concatenate([np.zeros(len(self.target_data)), np.ones(len(tmp_context))])
        class_weight = {0:len(label[label==1]), 1:len(label[label==0])}
        return data, label, class_weight
            
        
    def augmentation_generator_autoencoder(self):
        '''
        keras learning function
        '''
        while True:
            for j in range(self.target_data.shape[0]):
                tmp_context = []
                y = np.repeat(np.expand_dims(self.train_data[j], axis=0), self.k, axis=0)
                if self.da_type=='norm':
                    tmp_context += self.gaussian_noise(self.target_data[j], self.k)
                elif self.da_type == 'inter':
                    tmp_context += self.interpolation_augmentation(self.target_data[j])
                elif self.da_type == 'extra':
                    tmp_context += self.extrapolation_augmentation(self.target_data[j])
                elif self.da_type == 'hard_extra':
                    yield self.hard_extrapolation_augmentation(self.target_data[j]), np.expand_dims(self.train_data[j], axis=0)[:, ::-1, :]
                
                yield np.array(tmp_context), y[:, ::-1, :]
    
    def augmentation_autoencoder(self):
        '''
        keras learning function
        '''
        x_train = []
        y_train = []
        for j in range(self.target_data.shape[0]):
            tmp_context = []
            y = np.repeat(np.expand_dims(self.train_data[j], axis=0), self.k, axis=0)
            if self.da_type=='norm':
                tmp_context += self.gaussian_noise(self.target_data[j], self.k)
            elif self.da_type == 'inter':
                tmp_context += self.interpolation_augmentation(self.target_data[j])
            elif self.da_type == 'extra':
                tmp_context += self.extrapolation_augmentation(self.target_data[j])
            # elif self.da_type == 'hard_extra':
            #     yield self.hard_extrapolation_augmentation(self.target_data[j]), np.expand_dims(self.train_data[j], axis=0)[:, ::-1, :]
            
            # yield np.array(tmp_context), y[:, ::-1, :]
            if self.da_type == 'hard_extra':
                x_train += self.hard_extrapolation_augmentation(self.target_data[j]).tolist()
                y_train += np.expand_dims(self.train_data[j], axis=0)[:, ::-1, :].tolist()          
            else:
                x_train += tmp_context
                y_train += y[:, ::-1, :].tolist()

        return np.array(x_train), np.array(y_train)
                
    def set_generator(self, da_type='norm', k=10, train_data=None):
        '''
        genrator setting function
        Arg:
            da_type (str): data augmentation type
            k (int): nearest neighbor k
            train_data (numpy.array): train data
        '''
        self.train_data = train_data
        self.da_type=da_type
        self.k=10
        self.augmentation_init(self.da_type, self.k)