import sys
sys.path.append("..")
import pandas as pd
import numpy as np
import math
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from XJTU_Related.XJTU_Dataset import XJTUData
from torch.utils.data import DataLoader

# load each file:
def load_file_acc(file_path, id, bearing_num):
    file_acc_df = pd.read_csv(file_path)
    # 增加一列用来在tsfresh中作为分类提取数据的标签
    file_acc_df["file_index"] = id+1
    file_acc_df.set_index("file_index", inplace=True)
    file_acc_df["id"] = int(str(bearing_num)+f"{str(id+1).zfill(4)}")
    file_acc_df["file_time"] = [i for i in range(0, len(file_acc_df))]
    return file_acc_df


# 读取一个bearing中的文件
def get_bearing_acc(folder_path, bearing_num):
    file_name_ls = os.listdir(folder_path)
    file_num = len(file_name_ls)  # Bearing文件夹中有多少个acc文件

    # 遍历读取所有acc文件,并将其组合在一起
    acc_ls = []
    for id in range(0, file_num):
        acc_file_path = folder_path + "/" + f"{str(id+1)}.csv"
        file_acc_df = load_file_acc(acc_file_path, id, bearing_num)
        acc_ls.append(file_acc_df)
    acc_df = pd.concat(acc_ls, axis=0, ignore_index=False)  # df的id是从1开始的,从'load_file_acc'中,id会加1

    return acc_df, file_num


# calculate rul
def rul_calculate(file_num, bearing_num):
    rul_ls = []
    for i in range(1, int(file_num)+1):
        rul_time = (file_num - i)  
        rul_ls.append(rul_time)
    rul_dataframe = pd.DataFrame(rul_ls, columns=['RUL'])
    rul_dataframe["id"] = [int(str(bearing_num)+f"{str(id+1).zfill(4)}") for id in range(0, file_num)]
    rul_dataframe.set_index("id", inplace=True)
    return rul_dataframe 


def data_visual_rul(rul_dataframe, bearing_acc_dataframe, bearing_name):    
    plt.figure(figsize=(24, 6))
    # plot1 - rul
    x = list(range(1, len(rul_dataframe)+1))
    plt.subplot(131)
    plt.plot(x, rul_dataframe.values)
    plt.title("RUL")
    plt.xlabel("sample file number")
    plt.ylabel("RUL 10(s)")

    # plot2 - Horizontal_vibration_signals
    plt.subplot(132)
    plt.plot(bearing_acc_dataframe[["Horizontal_vibration_signals"]])
    plt.title("Horizontal_vibration_signals")
    plt.xlabel("sample file number")
    plt.ylabel("vibration signal")

    # plot3 - Vertical_vibration_signals
    plt.subplot(133)
    plt.plot(bearing_acc_dataframe[["Vertical_vibration_signals"]])
    plt.title("Vertical_vibration_signals")
    plt.xlabel("sample file number")
    plt.ylabel("vibration signal")
    plt.tight_layout()
    plt.savefig('/content/drive/MyDrive/picture_XJTU/acc&rul_picture/acc&rul_{}.jpg'.format(bearing_name))


# 加载训练集/验证集数据
def data_load(root_dir, bearing_data_set, flag="train"):
    if flag == "train":
        print("------------------------------ start load training data ------------------------------")
        acc_dataframe_ls, rul_dataframe_ls = [], []
        file_num_ls = []
        for bearing_num, bearing_name in enumerate(bearing_data_set):
            bearing_num += 1
            folder_path = root_dir + "/" + bearing_name

            # load acc_data from bearing
            bearing_acc_dataframe, bearing_file_num = get_bearing_acc(folder_path, bearing_num) 
            print("{0} have {1} acceleration files".format(bearing_name, bearing_file_num))
            acc_dataframe_ls.append(bearing_acc_dataframe)
            file_num_ls.append(bearing_file_num) 

            # load rul from bearing
            bearing_rul_dataframe = rul_calculate(bearing_file_num, bearing_num)   
            rul_dataframe_ls.append(bearing_rul_dataframe)

            # picture of bearing_acc_dataframe and bearing_rul_dataframe
            #data_visual_rul(bearing_rul_dataframe, bearing_acc_dataframe, bearing_name)
        
        train_x_dataframe = pd.concat(acc_dataframe_ls, axis=0, ignore_index=False)
        train_y_dataframe = pd.concat(rul_dataframe_ls, axis=0, ignore_index=False)

        return train_x_dataframe, train_y_dataframe, file_num_ls

    else:
        print("------------------------------ start load test data ------------------------------")
        bearing_name = bearing_data_set[0]
        bearing_num = 5    # train_set in each condition have 2 bearing dataset
        folder_path = root_dir + "/" + bearing_name

        # load acc_data from bearing
        bearing_acc_dataframe, bearing_file_num = get_bearing_acc(folder_path, bearing_num) 
        file_num_ls = [bearing_file_num]
        print("{0} have {1} acceleration files".format(bearing_name, bearing_file_num))

        # load rul from bearing
        bearing_rul_dataframe = rul_calculate(bearing_file_num, bearing_num)

        # picture of bearing_acc_dataframe and bearing_rul_dataframe
        #data_visual_rul(bearing_rul_dataframe, bearing_acc_dataframe, bearing_name)

        test_x_dataframe, test_y_dataframe = bearing_acc_dataframe, bearing_rul_dataframe

        return test_x_dataframe, test_y_dataframe, file_num_ls



# ------------------------ 数据加载与处理函数 --------------------------
def get_xjtu_data_(pre_process_type, root_dir, train_bearing_data_set, test_bearing_data_set, STFT_window_len, STFT_overlap_num, window_length, validation_rate):
    # 加载原始数据并组合,绘制rul与水平&纵向加速度图
    train_x_dataframe, train_y_dataframe, train_file_num_ls = data_load(root_dir, train_bearing_data_set, flag="train")
    test_x_dataframe, test_y_dataframe, test_file_num_ls = data_load(root_dir, test_bearing_data_set, flag="test")

    # --------------------------------------- Vibration -------------------------------------------------------------------------------
# 计算训练集和测试集的合力vibration, 再将训练集根据文件分为两个
    train_x_dataframe = pd.read_csv('./XJTU_Related/train_x_vibration', index_col=0)

    test_x_dataframe = pd.read_csv('./XJTU_Related/test_x_vibration', index_col=0)

    # **standarlization**
    vibration_train_x_np = train_x_dataframe[["vibration"]].values
    vibration_test_x_np = test_x_dataframe[["vibration"]].values

    del train_x_dataframe
    del test_x_dataframe

    vibration_mean = np.mean(vibration_train_x_np, axis=0).reshape(-1, 1)
    vibration_std = np.std(vibration_train_x_np, axis=0).reshape(-1, 1)

    stand_train_vibration_feature = (vibration_train_x_np - vibration_mean) / vibration_std
    stand_test_vibration_feature = (vibration_test_x_np - vibration_mean) / vibration_std

    del vibration_train_x_np
    del vibration_test_x_np
    del vibration_mean
    del vibration_std

    # **split based file_num**
    split_ratio = 40
    train_split_ls = []
    train_file_cycle_ls = [file * int(32768 / split_ratio) for file in train_file_num_ls]
    for i in range(0, len(train_file_num_ls)):
        for j in range(train_file_cycle_ls[i]):
            start_index = i * 32768
            split_file_feature_train = stand_train_vibration_feature[
                                       start_index + split_ratio * j:start_index + split_ratio * (j + 1), :]
            split_file_feature_train = split_file_feature_train.transpose(1, 0)
            train_split_ls.append(split_file_feature_train)

    train_file_splited_feature = np.concatenate(train_split_ls, axis=0)
    del stand_train_vibration_feature
    del split_file_feature_train
    del train_split_ls

    test_split_ls = []
    test_file_cycle_ls = [file * int(32768 / split_ratio) for file in test_file_num_ls]
    for i in range(0, test_file_cycle_ls[0]):
        split_file_feature_test = stand_test_vibration_feature[split_ratio * i:split_ratio * (i + 1), :]
        split_file_feature_test = split_file_feature_test.transpose(1, 0)
        test_split_ls.append(split_file_feature_test)

    test_file_splited_feature = np.concatenate(test_split_ls, axis=0)
    del stand_test_vibration_feature
    del split_file_feature_test
    del test_split_ls

    # **window_length**
    # window_length_train
    train_y_array = train_y_dataframe.values
    for i in range(len(train_y_array)):
        added_split_train_y = np.linspace(0, 1, int(32768 / split_ratio) - 1)
        train_y_array = np.insert(train_y_array, i + 1, added_split_train_y + train_y_array[i])  # TODO 为什么全部自动变成整数？

    train_y_array = train_y_array[:, np.newaxis]
    win_train_x_ls, win_train_y_ls = [], []

    for i in range(0, len(train_file_cycle_ls)):
        split_trian_x_array = train_file_splited_feature[0:train_file_cycle_ls[i], :]
        split_train_y_array = train_y_array[0:train_file_cycle_ls[i], :]
        train_file_splited_feature = train_file_splited_feature[train_file_cycle_ls[i]:, :]
        train_y_array = train_y_array[train_file_cycle_ls[i]:, :]

        seg_fea_ls = []
        seg_rul_ls = []
        for i in range(0, len(split_train_y_array) - window_length + 1):
            seg_fea = split_trian_x_array[i:i + window_length, :]
            seg_fea = np.expand_dims(seg_fea, axis=0)
            seg_fea_ls.append(seg_fea)
            seg_rul = split_train_y_array[i:i + window_length, :]
            seg_rul = seg_rul.transpose(1, 0)
            seg_rul_ls.append(seg_rul)
        win_train_x = np.concatenate(seg_fea_ls, axis=0)
        win_train_y = np.concatenate(seg_rul_ls, axis=0)

        win_train_x_ls.append(win_train_x)
        win_train_y_ls.append(win_train_y)

    del seg_fea_ls
    del seg_fea
    del seg_rul_ls
    del seg_rul
    del win_train_x
    del win_train_y
    del train_file_splited_feature
    del split_train_y_array
    del split_trian_x_array
    del train_y_dataframe
    del train_y_array
    del train_file_num_ls

    X_train = np.concatenate(win_train_x_ls, axis=0)
    y_train = np.concatenate(win_train_y_ls, axis=0)
    del win_train_x_ls
    del win_train_y_ls

    print(
        "the shape of training set is {0} and the shape of train label is {1}".format(X_train.shape, y_train.shape))

    # window_length_test
    test_y_array = test_y_dataframe.values
    for i in range(len(test_y_array)):
        added_split_test_y = np.linspace(0, 1, int(32768 / split_ratio) - 1)
        test_y_array = np.insert(test_y_array, i + 1, added_split_test_y + test_y_array[i])  # TODO 为什么全部自动变成整数？

    test_y_array = test_y_array[:, np.newaxis]

    seg_fea_ls = []
    seg_rul_ls = []
    for i in range(0, len(test_y_array) - window_length + 1):
        seg_fea = test_file_splited_feature[i:i + window_length, :]
        seg_fea = np.expand_dims(seg_fea, axis=0)
        seg_fea_ls.append(seg_fea)
        seg_rul = test_y_array[i:i + window_length, :]
        seg_rul = seg_rul.transpose(1, 0)
        seg_rul_ls.append(seg_rul)

    del test_y_dataframe
    del test_y_array
    del test_file_splited_feature
    del seg_fea
    del seg_rul

    test_X = np.concatenate(seg_fea_ls, axis=0)
    test_y = np.concatenate(seg_rul_ls, axis=0)

    del seg_fea_ls
    del seg_rul_ls

    print("the shape of test_X is {0} and the shape of test_y is {1}".format(test_X.shape, test_y.shape))

    # train_validation_split
    train_X, vali_X, train_y, vali_y = train_test_split(X_train, y_train, test_size=validation_rate, random_state=42)
    print("the shape of train_X is {0} and the shape of train_y is {1}".format(train_X.shape, train_y.shape))
    print("the shape of vali_X is {0} and the shape of vali_y is {1}".format(vali_X.shape, vali_y.shape))

    return train_X, train_y, vali_X, vali_y, test_X, test_y

def da_get_xjtu_data_(pre_process_type, root_dir, train_bearing_data_set, test_bearing_data_set, STFT_window_len, STFT_overlap_num, window_length, validation_rate):
    s_train_X, s_train_y, s_vali_X, s_vali_y, t_test_X, t_test_y = get_xjtu_data_(
        pre_process_type="Vibration",
        root_dir='./XJTU/XJTU-SY_Bearing_Datasets/35Hz12kN',
        train_bearing_data_set=["Bearing1_2", "Bearing1_3", "Bearing1_4", "Bearing1_5"],
        test_bearing_data_set=["Bearing1_1"],
        STFT_window_len=256,
        STFT_overlap_num=32,
        window_length=window_length,
        validation_rate=0.1)

    t_train_X, t_vali_X, t_train_y, t_vali_y = s_train_X, s_vali_X, s_train_y, s_vali_y

    return s_train_X, t_train_X, s_vali_X, t_vali_X, t_train_y, t_vali_y, t_test_X, t_test_y

if __name__ == '__main__':

    train_X, vali_X, test_X, train_y, vali_y, test_y = get_xjtu_data_(
        pre_process_type="Vibration",
        root_dir='./XJTU/XJTU-SY_Bearing_Datasets/35Hz12kN',
        train_bearing_data_set=["Bearing1_2", "Bearing1_3", "Bearing1_4", "Bearing1_5"],
        test_bearing_data_set=["Bearing1_1"],
        STFT_window_len=256,
        STFT_overlap_num=32,
        window_length=32,
        validation_rate=0.1)

    train_data = XJTUData(train_X, train_y)
    train_loader = DataLoader(dataset=train_data,
                              batch_size=32,
                              shuffle=True,
                              num_workers=0,
                              drop_last=False)

    vali_data = XJTUData(vali_X, vali_y)
    vali_loader = DataLoader(dataset=vali_data,
                             batch_size=32,
                             shuffle=False,
                             num_workers=0,
                             drop_last=False)

    test_data = XJTUData(test_X, test_y)
    test_loader = DataLoader(dataset=test_data,
                             batch_size=32,
                             shuffle=False,
                             num_workers=0,
                             drop_last=False)

    input_fea = test_X.shape[-1]



