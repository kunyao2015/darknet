# �˴����data�ļ���ͬĿ¼
import glob
path = 'kitti_data/'
def generate_train_and_val(image_path, txt_file):
    with open(txt_file, 'w') as tf:
        for jpg_file in glob.glob(image_path + '*.png'):
            tf.write(jpg_file + '\n')
generate_train_and_val(path + 'train_images/', path + 'train.txt') # ���ɵ�train.txt�ļ�����·��
# generate_train_and_val(path + 'val_images/', path + 'val.txt') # ���ɵ�val.txt�ļ�����·��