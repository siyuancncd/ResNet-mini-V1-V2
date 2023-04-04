import os

image_types = ('.jpg', '.png', '.jpeg', '.bmp', '.tif', 'tiff')

#返回根目录下所有有效的图片路径数据集
def list_images(basePath, contains = None):
    return list_files(basePath, validExts = image_types, contains = contains)

def list_files(basePath, validExts = None, contains = None):
    for (rootDir, dirNames, filenames) in os.walk(basePath): #basePath是top，是要遍历的目录
        for filename in filenames:
            #此时filename是str
            #检测文件名，如果没有文件名则继续查找，如果有则继续下面的代码
            if contains is not None and filename.find(contains) == -1: #find()方法检测字符串中有没有()里的，如果有则返回引锁值，没用则返回-1
                continue
            #ext是后缀，.rfind()返回（）中字符串最后出现的位置
            ext = filename[filename.rfind('.'):].lower() #通过确定.的位置从而确定当前文件夹的文件扩展名
                                                         #是切片操作
            if validExts is None or ext.endswith(validExts):
                imagePath = os.path.join(rootDir, filename)
                yield imagePath


