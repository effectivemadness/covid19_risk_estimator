

import glob
import os
import shutil
img_path = 'C:/Users/madcat/darknet-crowdhuman/crowdhuman_train/Images'
anno_path = 'C:/Users/madcat/darknet-crowdhuman/crowdhuman_train/anno_mask_to_bad'
new_path = 'C:/Users/madcat/darknet-crowdhuman/crowdhuman_train/img_mask_to_bad'
dataset_file = open('mask_to_bad.txt','r')
dataset_file_list = dataset_file.readlines()
print(dataset_file_list)
for item in dataset_file_list:
    file_name = os.path.basename(item[0:-1])
    shutil.copy(os.path.join(img_path,file_name), new_path)
    shutil.copy(os.path.join(anno_path,file_name[0:-3]+'txt'), new_path)

# for i in range(1,679):

#     anno_file = open('mask/mask_'+str(i)+'.txt')
#     anno_new = open('mask_new/mask_'+str(i)+'.txt', 'w')
#     while True:
#         line = anno_file.readline()
#         splited_line = line.split()
#         if splited_line == []: break
        
#         new_line = str(int(splited_line[0])+1) + " " + splited_line[1] + " " + splited_line[2] + " " + splited_line[3]  + " " + splited_line[4]
#         print(new_line,file=anno_new)
#         if not line: break
#         print(splited_line)