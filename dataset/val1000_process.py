import os
import shutil
import pandas as pd
# 读取csv文件
df = pd.read_csv('val_1000.csv')
# 源文件夹路径
src_folder = 'D:\data\ImageNet\ILSVRC2012_img_val'
# 目标文件夹路径
dst_folder = 'dataset'
# 创建目标文件夹
if not os.path.exists(dst_folder):
    os.mkdir(dst_folder)
# 遍历csv文件中的每一行
for index, row in df.iterrows():
    # 获取文件名和标签
    filename = row['filename']
    label = row['label']
    # 在源文件夹中搜索与文件名一致的图像
    for root, dirs, files in os.walk(src_folder):
        for file in files:
            if file == filename:
                # 获取该图像所在的文件夹名称
                folder_name = os.path.basename(root)
                # 源文件路径
                src_path = os.path.join(root, file)
                # 目标文件路径
                dst_path = os.path.join(dst_folder, filename)
                # 复制文件
                shutil.copy(src_path, dst_path)
                # 输出提示信息
                print(f'Copied {filename} from {folder_name} to {dst_folder}')

print('Done!')
