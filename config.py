import os

home_path = '/home/ran'
root_path = home_path+'/HDD/Data/ILSVRC/ILSVRC'
tfrecords_path = home_path+'/HDD/Data/ILSVRC/ILSVRC-TF'
otb_data_dir= home_path+'/Trails/A-10-run05'
seq='03'#
ver='1'#
code_seq = '/CODE/'+seq+'/'
image_seq='/CODE/'+seq+'/'+seq+'jpg'+'/'
res_seq='/RESULTS/TR_'+seq+'_RES/V'+ver+'/'#
 

#otb_data_dir = home_path+'/Data/Benchmark/OTB'
data_path_t = os.path.join(root_path, 'Data/VID/train')
data_path_v = os.path.join(root_path, 'Data/VID/val')
anno_path_t = os.path.join(root_path, 'Annotations/VID/train/')
anno_path_v = os.path.join(root_path, 'Annotations/VID/val/')

vid_info_t = './VID_Info/vid_info_train.txt'
vid_info_v = './VID_Info/vid_info_val.txt'
vidb_t = './VID_Info/vidb_train.pk'
vidb_v = './VID_Info/vidb_val.pk'




training_dir = "/content/drive/My Drive/data/sign_data/train"
training_csv = "/content/drive/My Drive/data/sign_data/train_data.csv"
testing_csv = "/content/drive/My Drive/data/sign_data/test_data.csv"
testing_dir = "/content/drive/My Drive/data/sign_data/test"
batch_size = 32
epochs = 20


#========================== data input ============================
min_queue_examples = 500
num_readers = 2
num_preprocess_threads = 8

z_exemplar_size = 127
x_instance_size = 255

is_limit_search = False
max_search_range = 200

is_augment = True
max_strech_x = 0.05
max_translate_x = 4
max_strech_z = 0.1
max_translate_z = 8

label_type= 0 # 0: overlap: 1 dist
overlap_thres = 0.7
dist_thre = 2
