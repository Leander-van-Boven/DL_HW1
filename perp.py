import os
from shutil import copyfile


for file in os.listdir('./data'):
    file_root = file.split('.')[0]
    print('./prep/%s' % file_root)
    os.makedirs('./prep/%s' % file_root, exist_ok=True)
    copyfile('./data/%s' % file, './prep/%s/%s' % (file_root, 'img.png'))