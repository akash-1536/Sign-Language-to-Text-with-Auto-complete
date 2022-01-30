import os
import cv2

#devoloping training dataet
path="dataset_massey_univ/data/raw_data/train"
fin_path="dataset_massey_univ/data/preprocessed/train"
dir_list = os.listdir(path)

for directory in dir_list:
    path1=path+"/"+directory
    dir_list1=os.listdir((path1))
    fin_path2 = fin_path + "/" + directory
    if not os.path.exists(fin_path2):
        os.makedirs(fin_path2)
    count=len(os.listdir(fin_path2))
    for i in range(2):
        for img in dir_list1:
            path2=path1+"/"+img
            pre_img=cv2.imread(path2)

            minValue=70

            # dim=(310,310)
            # resized=cv2.resize(pre_img, dim, interpolation = cv2.INTER_AREA)
            img_height = pre_img.shape[0]
            img_width = pre_img.shape[1]
            top = 0
            bottom = 0
            left = 0
            right = 0
            if img_height > img_width:
                left = (img_height - img_width) // 2
                right = img_height - img_width - left
            elif img_height < img_width:
                top = (img_width - img_height) // 2
                bottom = img_width - img_height - top
            resized1 = cv2.copyMakeBorder(pre_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0])
            dim = (310, 310)
            resized2 = cv2.resize(resized1, dim, interpolation=cv2.INTER_AREA)
            if(directory=='0'):
                dim = (310, 310)
                resized2 = cv2.resize(pre_img, dim, interpolation=cv2.INTER_AREA)

            gray = cv2.cvtColor(resized2, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 2)
            th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            ret, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


            filename=fin_path2+"/"+str(count)+".jpg"
            #print(filename)
            cv2.imwrite(filename,res)
            count=count+1









#devoloping testing dataset
path="dataset_massey_univ/data/raw_data/test"
fin_path="dataset_massey_univ/data/preprocessed/test"
dir_list = os.listdir(path)



for directory in dir_list:
    path1=path+"/"+directory
    dir_list1=os.listdir((path1))
    fin_path2 = fin_path + "/" + directory
    if not os.path.exists(fin_path2):
        os.makedirs(fin_path2)
    count=len(os.listdir(fin_path2))
    for i in range(3):
        for img in dir_list1:
            path2=path1+"/"+img
            pre_img=cv2.imread(path2)

            minValue=70

            # dim=(310,310)
            # resized=cv2.resize(pre_img, dim, interpolation = cv2.INTER_AREA)
            img_height = pre_img.shape[0]
            img_width = pre_img.shape[1]
            top = 0
            bottom = 0
            left = 0
            right = 0
            if img_height > img_width:
                left = (img_height - img_width) // 2
                right = img_height - img_width - left
            elif img_height < img_width:
                top = (img_width - img_height) // 2
                bottom = img_width - img_height - top
            resized1 = cv2.copyMakeBorder(pre_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0])
            dim = (310, 310)
            resized2 = cv2.resize(resized1, dim, interpolation=cv2.INTER_AREA)
            if (directory == '0'):
                dim = (310, 310)
                resized2 = cv2.resize(pre_img, dim, interpolation=cv2.INTER_AREA)

            gray = cv2.cvtColor(resized2, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 2)
            th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            ret, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


            filename=fin_path2+"/"+str(count)+".jpg"
            cv2.imwrite(filename,res)
            count=count+1