import cv2
with open('data/ground_truth.txt', 'r') as file_to_read:
    i = 0
    lines = file_to_read.readline()
    pic_num, var1, var2, var3 = [i for i in lines.split()]
    first_pic = cv2.imread(pic_num+".jpg")
    while i<3:
        lines = file_to_read.readline()
        if not lines:
            break
        pic_num, var1,var2,var3 = [i for i in lines.split()]# 整行读取数据
        second_pic = cv2.imread(pic_num+".jpg")
        """some code here"""
        first_pic = second_pic
        i += 1
        print([pic_num, var1,var2,var3])
