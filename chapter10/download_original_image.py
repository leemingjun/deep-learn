"""
    首先，下载原始图片文件的地址，解压后得到一个文本文件（如fall11_urls.txt），内容如：

    n02084071_1	http://farm1.static.flickr.com/164/358144227_01e5544b79.jpg
    n02084071_7	http://www.pantherkut.com/wp-content/uploads/2007/04/2.jpg

    第一部分是图像的标识，如‘n02084071_1’，下划线之前的部分是同义词标识，下划线之后是
    该同义词中第几个图片。

    第二部分是该图像的URL地址。

    注意：本例子中没有采用多线程处理。实际上下载大约 130 万个图片，大概需要下载几天
    建议采用过线程处理。如何实现多线程下载呢？
"""


import os
import urllib.request

# 数据及保存的本地文件地址
data_dir = './data/imagenet/'

# 包含图像原始URL 列表的文本文件
# 如果要实现多线程下载，最简单的就是将 'fall11_urls.txt' 分成几个部分
# 然后，分别执行本 python 脚本
# 其他的方法，就是改写如下脚本，将 'fall11_urls.txt' 中包含的下载地址
# 分片，然后，每个线程处理处理一个分片
url_file = os.path.join(data_dir, 'fall11_urls.txt')
with open(url_file) as f:
    for line in f:
        if line:
            parts = line.strip().split('\t')
            assert len(parts) == 2
            # 读取到图像的标识，形如 n02084071_1、n02084071_7
            # 读取下划线之前的部分，如 "n02084071"
            synset = parts[0][0: parts[0].index('_')]

            # 生成图片保存的路径，如 "${data_dir}/images/${synset}/"
            img_path = os.path.join(data_dir, 'images', synset)
            # 确保相关目录、及其子目录存在
            os.makedirs(img_path, exist_ok=True)

            img_url = parts[1]
            # 图片保存的本地路径
            img_file = os.path.join(
                img_path, "{}.jpg".format(parts[0]))

            try:
                # 从 img_url 下载图片 保存到 img_file
                urllib.request.urlretrieve(img_url, img_file)
                print("下载: {} 保存到: {}".format(img_url, img_file))
            except:
                print("下载: {} 失败！".format(img_url))
