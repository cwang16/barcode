import cv2, os
import numpy as np
import glob, math
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from functools import reduce
import operator
import math
import json

#prediction_labeltxt_path = "/media/lab/8TB_EXT4/bar_code/steps/crop/pytorch-rotation-decoupled-detector-master/aws_model/dir_save/submission"
prediction_labeltxt_path = "/media/lab/8TB_EXT4/bar_code/steps/crop/pytorch-rotation-decoupled-detector-master" \
                           "/aws_model/dir_save/submission"
correspond_test_image_path = "/media/lab/8TB_EXT4/bar_code/steps/crop/pytorch-rotation-decoupled-detector-master" \
                             "/images/test"
result_dir = "/media/lab/8TB_EXT4/bar_code/steps/crop/pytorch-rotation-decoupled-detector-master/" \
             "DOTA_devkit-master/test_result"
ground_truth_label_dir = "/media/lab/8TB_EXT4/bar_code/datasets/original_datasets/all_in_one/imgs_by_train_val_test_raw_final/test"
mapping_file_dir = "/media/lab/8TB_EXT4/bar_code/datasets/original_datasets/all_in_one/imgs_by_train_val_test_raw_final/test_DBsrc"
cropped_path = "/media/lab/8TB_EXT4/bar_code/datasets/original_datasets/all_in_one/paired/cropped"
#'''

def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def sort_points_clockwise(coords):
    #coords = [(0, 1), (1, 0), (1, 1), (0, 0)]
    center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), coords), [len(coords)] * 2))
    #print(sorted(coords, key=lambda coord: (-135 - math.degrees(
    #    math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360))
    return sorted(coords, key=lambda coord: (-135 - math.degrees(math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360)
# this function is base on post at https://goo.gl/Q92hdp
def crop_rect(img, rect):
    # get the parameter of the small rectangle
    center = rect[0]
    size = rect[1]
    angle = rect[2]
    center, size = tuple(map(int, center)), tuple(map(int, size))


    # get row and col num in img
    rows, cols = img.shape[0], img.shape[1]

    M = cv2.getRotationMatrix2D(center, angle, 1)
    img_rot = cv2.warpAffine(img, M, (cols, rows))
    out = cv2.getRectSubPix(img_rot, size, center)

    return out, img_rot

def read_label_me_json(image_path, ground_truth_label_dir, blk_img):
    label_name = image_path.split("/")[-1].replace("jpg", "json")
    ground_truth_label_path = os.path.join(ground_truth_label_dir,label_name)

    with open(ground_truth_label_path, "rb") as fin:
        content = json.load(fin)
        coords = content["shapes"][0]["points"]
        point1, point2, point3, point4 = coords[0], coords[1],coords[2],coords[3]
        roi_corners = np.array([[(point1[0], point1[1]), (point2[0], point2[1]), (point3[0], point3[1]),(point4[0], point4[1])]], dtype=np.int32)
        cv2.fillConvexPoly(blk_img, roi_corners, (255, 255, 255))

        coord_tuple = [(point1[0],point1[1]),(point2[0],point2[1]),(point3[0],point3[1]),(point4[0],point4[1])]
        clockwise_coord = sort_points_clockwise(coord_tuple)
        p1_x = clockwise_coord[0][0]
        p1_y = clockwise_coord[0][1]
        p2_x = clockwise_coord[1][0]
        p2_y = clockwise_coord[1][1]
        p3_x = clockwise_coord[2][0]
        p3_y = clockwise_coord[2][1]
        p4_x = clockwise_coord[3][0]
        p4_y = clockwise_coord[3][1]
        cnt = np.array([[(p1_x, p1_y), (p2_x, p2_y), (p3_x, p3_y),(p4_x, p4_y)]], dtype=np.int32)
        angle = cv2.minAreaRect(cnt)[-1] # it has nothing to do clockwise, just becasue of the output
        # see more: https://stackoverflow.com/questions/15956124/minarearect-angles-unsure-about-the-angle-returned
        angle = abs(angle)
        if angle >45:
            angle=90-angle
        return blk_img, roi_corners, angle

def mapping_name (img_path, mapping_file_dir):
    img_n = img_path.split("/")[-1].replace(".jpg", "_")
    for mapping_file_name in os.listdir(mapping_file_dir):
        if mapping_file_name.startswith(img_n) and mapping_file_name.endswith(".jpg"):
            return mapping_file_name.split("_")[-1].replace(".jpg", "") # last element is the DBsrc




def four_point_crop (img_path, coord_tuple, coord_list, saved_fig_path, cropped_path):
    try:
        img = cv2.imread(img_path)
        img_cp = cv2.imread(img_path)
        img_cp_for_crop_saving = cv2.imread(img_path)
        print ("current: ", img_path)
        height, width, channels = img_cp.shape
        blank_image = np.zeros((height, width, 3), np.uint8)
        blank_image_cp_for_ground_truth = np.zeros((height, width, 3), np.uint8)
        clockwise_coord = sort_points_clockwise(coord_tuple)
        # assume coord is a list with 8 float values, the points of the rectangle area should
        # have be clockwise
        x1, y1, x2, y2, x3, y3, x4, y4 = coord_list
        cnt = np.array([[[x1, y1]],
                        [[x2, y2]],
                        [[x3, y3]],
                        [[x4, y4]]
                        ])
        # cv2.drawContours(img, [cnt], 0, (128, 255, 0), 3)
        # find the rotated rectangle enclosing the contour
        # rect has 3 elments, the first is rectangle center, the second is
        # width and height of the rectangle and the third is the rotation angle
        rect = cv2.minAreaRect(cnt)
        print("rect: {}".format(rect))
        # convert rect to 4 points format
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        print("bounding box: {}".format(box))

        # draw the roated rectangle box in the image
        cv2.drawContours(img, [box], 0, (0, 255, 0), 6)
        cv2.drawContours(img_cp, [box], 0, (0, 255, 0), 6) # draw prediction in green
        # crop the rotated rectangle from the image
        print("current img:", img_path)
        im_crop, img_rot = crop_rect(img, rect)
        im_crop_saving, img_rot_saving = crop_rect(img_cp_for_crop_saving, rect)
        if im_crop.shape[0] > im_crop.shape[1]: # height> width, then rotated 90degree
            im_crop = cv2.rotate(im_crop, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
        #save cropped img
        if im_crop_saving.shape[0] > im_crop_saving.shape[1]: # height> width, then rotated 90degree
            im_crop_saving = cv2.rotate(im_crop_saving, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
        saved_crop_path = os.path.join(cropped_path, img_path.split("/")[-1])
        cv2.imwrite(saved_crop_path, im_crop_saving)
        # print("size of original img: {}".format(img.shape))
        # print("size of rotated img: {}".format(img_rot.shape))
        # print("size of cropped img: {}".format(im_crop.shape))

        # get prediction black and white image
        # roi_corners = np.array([[(10, 10), (300, 300), (10, 300)]], dtype=np.int32)
        roi_corners = cnt
        cv2.fillConvexPoly(blank_image, roi_corners, (255, 255, 255))


        # get ground truth black and white image
        blk_white_ground_truth, ground_truth_box, ground_truth_angle = read_label_me_json(img_path, ground_truth_label_dir, blank_image_cp_for_ground_truth)
        cv2.drawContours(img_cp, ground_truth_box, 0, (255, 0, 0), 6) # ground truth box is red
        intersection = np.logical_and(blk_white_ground_truth, blank_image)  # (GROUNT TRUTH, PREDICTION )
        union = np.logical_or(blk_white_ground_truth, blank_image)
        inter = np.sum(intersection)
        uni = np.sum(union)
        iou_score = "iou:" + str(inter / uni)
        union_count = "union" + str(uni)
        inter_count = "inter" + str(inter)

        DBsrc = mapping_name(img_path, mapping_file_dir)

        plt.figure(figsize=(20, 15))
        ax1 = plt.subplot(231)
        ax1.set_xlabel('cropped_box', fontsize=17)
        ax1.imshow(im_crop, cmap='gray')

        ax2 = plt.subplot(232)
        ax2.set_xlabel('original contour, Box angle for above img:' + str(int(abs(ground_truth_angle))) + "°", fontsize=17)
        ax2.imshow(img, cmap='gray')

        ax3 = plt.subplot(233)
        ax3.set_xlabel('rotated image', fontsize=17)
        ax3.imshow(img_rot, cmap='gray')

        ax4 = plt.subplot(234)
        ax4.set_xlabel('prediction(green last img)', fontsize=17)
        ax4.imshow(blank_image, cmap='gray')

        ax5 = plt.subplot(235)
        ax5.set_xlabel('ground truth(red last img)', fontsize=17)
        ax5.imshow(blk_white_ground_truth, cmap='gray')


        #iou_png = cv2.imread("/media/lab/8TB_EXT4/bar_code/steps/crop/pytorch-rotation-decoupled-detector-master/DOTA_devkit-master/iou.png")
        ax6 = plt.subplot(236)
        ax6.set_xlabel(iou_score + "=" + inter_count + "/" + union_count, fontsize=17)
        ax6.imshow(img_cp, cmap='gray')

        plt.suptitle(DBsrc, fontsize=35)  # or plt.suptitle('Main title')
        #plt.show()
        plt.savefig(saved_fig_path)
        return inter / uni, DBsrc, ground_truth_angle
    except:
        print ("exceptionerror:", img_path)

'''
def save_four_point_crop_for_eval_metric(cnt):
    # save eval coordinate in the format of
    # https://github.com/rafaelpadilla/review_object_detection_metrics/tree/main/data/database/dets
    #rect = cv2.minAreaRect(cnt) # this is not work for out of border point
    saved_txt_for_eval_tool_path = os.path.join(result_dir, pred_cordination_txt.split("/")[-1].replace(".txt", ".txt"))
    file2write = open(saved_txt_for_eval_tool_path, 'w')
    file2write.write(max_for_eval_tool_coords_format)
    file2write.close()
'''
sum_iou = 0
counter = 0
counter_05 = 0
counter_075 = 0
counter_095 = 0
counter_DB1,counter_DB2,counter_DB3,counter_DB4,counter_DB5, counter_DB6 = 0,0,0,0,0,0
degree_DB1,degree_DB2,degree_DB3,degree_DB4,degree_DB5, degree_DB6 = 0,0,0,0,0,0
non_EAN13_list = ["72.txt", "80.txt", "89.txt", "109.txt", "143.txt"]
iou1,iou2,iou3,iou4,iou5,iou6 = 0,0,0,0,0,0

#'''
txt_prediction_list = glob.glob(prediction_labeltxt_path + "/*.txt")
for no_need in non_EAN13_list:
    txt_prediction_list.remove(os.path.join(prediction_labeltxt_path,no_need))
#'''
#txt_prediction_list = glob.glob(prediction_labeltxt_path + "/*.txt")

for pred_cordination_txt in txt_prediction_list:
    i = os.path.join(correspond_test_image_path, pred_cordination_txt.split("/")[-1].replace("txt", "jpg"))
    image_p = os.path.join(correspond_test_image_path, pred_cordination_txt.split("/")[-1].replace("txt", "jpg"))
    image = cv2.imread(image_p)
    pred_txt = open(pred_cordination_txt, 'r')
    Lines = pred_txt.readlines()
    count = 0
    # Strips the newline character
    max_area = 0
    max_area_pts = np.ones
    max_coordinate_np = np.ones
    max_coords_tuple_list = []
    max_coords_list = []
    max_for_eval_tool_coords_format = []
    for line in Lines:
        count += 1
        line_element_list = line.split(" ")
        point1 = [int(round(float(line_element_list[0]))) , int(round(float(line_element_list[1])))]
        point2 = [int(round(float(line_element_list[2]))) , int(round(float(line_element_list[3])))]
        point3 = [int(round(float(line_element_list[4]))) , int(round(float(line_element_list[5])))]
        point4 = [int(round(float(line_element_list[6]))) , int(round(float(line_element_list[7])))]
        confidence_score = float(line_element_list[-1])
        print (point1, "  ", point2, "  ", point3, "  ", point4, "  ", confidence_score)
        pts = np.array([point1, point2,
                        point3, point4],
                       np.int32)
        pts = pts.reshape((-1, 1, 2))
        isClosed = True
        # Blue color in BGR
        color = (0, 255, 0)
        # Line thickness of 2 px
        thickness = 2
        # Using cv2.polylines() method
        # Draw a Blue polygon with
        # thickness of 1 px
        coords=((point1[0],point1[1]),(point2[0],point2[1]),(point3[0],point3[1]),(point4[0],point4[1]))
        coords_np = np.array(([[[point1[0], point1[1]], [point2[0], point2[1]], [point3[0], point3[1]], [point4[0], point4[1]]]]), dtype=np.int32)
        coords_tuple_list = [(point1[0],point1[1]),(point2[0],point2[1]),(point3[0],point3[1]),(point4[0],point4[1])]
        coords_list = [point1[0], point1[1], point2[0], point2[1], point3[0], point3[1], point4[0], point4[1]]
        for_eval_tool_coords_format = "barcode" + " " + str(confidence_score) + " " + str(point1[0]) + " " + str(point1[1]) + " " + str(point2[0]) + " " + str(point2[1]) + " " + str(point3[0]) + " " + str(point3[1]) + " " + str(point4[0]) + " " + str(point4[1])
        #for_eval_tool_coords_format = "barcode" + " " + str(confidence_score) + " " + str(point1[0]) + " " + str(point1[1]) + " " + str(point3[0]) + " " + str(point3[1])

        '''
        polygon = Polygon(coords)
        polygon_area = polygon.area
        if max_area < polygon_area: #and confidence_score > 0.99:
            max_area = polygon_area
            max_area_pts = pts
            max_coordinate_np = coords_np
            max_coords_tuple_list = coords_tuple_list
            max_coords_list = coords_list
            max_for_eval_tool_coords_format = for_eval_tool_coords_format
        '''
        polygon = Polygon(coords)
        polygon_area = polygon.area
        if max_area < polygon_area and confidence_score == 1.000000000000:  # and confidence_score > 0.99:
            max_area = polygon_area
            max_area_pts = pts
            max_coordinate_np = coords_np
            max_coords_tuple_list = coords_tuple_list
            max_coords_list = coords_list
            max_for_eval_tool_coords_format = for_eval_tool_coords_format
            continue
        elif max_area < polygon_area and confidence_score > 0.9:
            max_area = polygon_area
            max_area_pts = pts
            max_coordinate_np = coords_np
            max_coords_tuple_list = coords_tuple_list
            max_coords_list = coords_list
            max_for_eval_tool_coords_format = for_eval_tool_coords_format
            continue
        elif max_area < polygon_area and confidence_score > 0.8:
            max_area = polygon_area
            max_area_pts = pts
            max_coordinate_np = coords_np
            max_coords_tuple_list = coords_tuple_list
            max_coords_list = coords_list
            max_for_eval_tool_coords_format = for_eval_tool_coords_format
            continue
        elif max_area < polygon_area and confidence_score > 0.7:
            max_area = polygon_area
            max_area_pts = pts
            max_coordinate_np = coords_np
            max_coords_tuple_list = coords_tuple_list
            max_coords_list = coords_list
            max_for_eval_tool_coords_format = for_eval_tool_coords_format
            continue
        elif max_area < polygon_area and confidence_score > 0.6:
            max_area = polygon_area
            max_area_pts = pts
            max_coordinate_np = coords_np
            max_coords_tuple_list = coords_tuple_list
            max_coords_list = coords_list
            max_for_eval_tool_coords_format = for_eval_tool_coords_format
            continue
        elif max_area < polygon_area and confidence_score > 0.5:
            max_area = polygon_area
            max_area_pts = pts
            max_coordinate_np = coords_np
            max_coords_tuple_list = coords_tuple_list
            max_coords_list = coords_list
            max_for_eval_tool_coords_format = for_eval_tool_coords_format
            continue
        elif max_area < polygon_area and confidence_score > 0.4:
            max_area = polygon_area
            max_area_pts = pts
            max_coordinate_np = coords_np
            max_coords_tuple_list = coords_tuple_list
            max_coords_list = coords_list
            max_for_eval_tool_coords_format = for_eval_tool_coords_format
            continue
        elif max_area < polygon_area and confidence_score > 0.3:
            max_area = polygon_area
            max_area_pts = pts
            max_coordinate_np = coords_np
            max_coords_tuple_list = coords_tuple_list
            max_coords_list = coords_list
            max_for_eval_tool_coords_format = for_eval_tool_coords_format
            continue
        elif max_area < polygon_area and confidence_score > 0.2:
            max_area = polygon_area
            max_area_pts = pts
            max_coordinate_np = coords_np
            max_coords_tuple_list = coords_tuple_list
            max_coords_list = coords_list
            max_for_eval_tool_coords_format = for_eval_tool_coords_format
            continue
        elif max_area < polygon_area and confidence_score > 0.1:
            max_area = polygon_area
            max_area_pts = pts
            max_coordinate_np = coords_np
            max_coords_tuple_list = coords_tuple_list
            max_coords_list = coords_list
            max_for_eval_tool_coords_format = for_eval_tool_coords_format
            continue
        elif max_area < polygon_area and confidence_score >= 0.0:
            max_area = polygon_area
            max_area_pts = pts
            max_coordinate_np = coords_np
            max_coords_tuple_list = coords_tuple_list
            max_coords_list = coords_list
            max_for_eval_tool_coords_format = for_eval_tool_coords_format
            continue

    image = cv2.polylines(image, [max_area_pts],
                              isClosed, color, thickness)
    result_image = os.path.join(result_dir, pred_cordination_txt.split("/")[-1].replace("txt", "jpg"))
    saved_fig_path = os.path.join(result_dir, pred_cordination_txt.split("/")[-1].replace(".txt", "_fig.jpg"))
    #cv2.imwrite(result_image, image)
    #if pred_cordination_txt.split("/")[-1] == "89.txt":
    # tutorial: https://gist.github.com/jdhao/1cb4c8f6561fbdb87859ac28a84b0201
    # description: Given an image and a 4 points in the image (no 3 points are co-linear).
    # Find the rotated rectangle enclosing the polygon formed by the 4 points and crop the rotated rectangle from the image.
    #save_four_point_crop_for_eval_metric(max_area_pts)
    iou, DBsrc, ground_truth_angle = four_point_crop(image_p, max_coords_tuple_list, max_coords_list, saved_fig_path, cropped_path)
    if iou >=0.5:
        counter_05 +=1
    if iou >=0.75:
        counter_075 +=1
    if iou >= 0.95:
        counter_095 += 1
    counter += 1
    sum_iou += iou
    print (DBsrc)
    if DBsrc =="InsubriaDatabase1":
        counter_DB1+= 1
        degree_DB1 += ground_truth_angle
        iou1+=iou
    elif DBsrc == "MuensterDatabase":
        counter_DB2 += 1
        degree_DB2 += ground_truth_angle
        iou2 += iou
    elif DBsrc == "zxingDatabase":
        counter_DB3 += 1
        degree_DB3 += ground_truth_angle
        iou3 += iou
    elif DBsrc == "InsubriaDatabase2":
        counter_DB4 += 1
        degree_DB4 += ground_truth_angle
        iou4 += iou
    elif DBsrc == "skkuinyougDatabase":
        counter_DB5 += 1
        degree_DB5 += ground_truth_angle
        iou5 += iou
    elif DBsrc == "openfoodDatabase":
        counter_DB6 += 1
        degree_DB6 += ground_truth_angle
        iou6 += iou
    else:
        print ("no database found")


print ("total image calculated for iou: ", counter)
print ("average_iou: ", sum_iou/counter)
print ("iou@0.5: ", counter_05)
print ("iou@0.75: ", counter_075)
print ("iou@0.95: ", counter_095)
print ("InsubriaDatabase1 iou: ", iou1/counter_DB1, " Total imgs: ", counter_DB1)
print ("MuensterDatabase iou: ", iou2/counter_DB2, " Total imgs: ", counter_DB2)
print ("zxingDatabase iou: ", iou3/counter_DB3, " Total imgs: ", counter_DB3)
print ("InsubriaDatabase2 iou: ", iou4/counter_DB4," Total imgs: ", counter_DB4)
print ("skkuinyougDatabase iou: ", iou5/counter_DB5, " Total imgs: ", counter_DB5)
print ("openfoodDatabase iou: ", iou6/counter_DB6, " Total imgs: ", counter_DB6)

print ("InsubriaDatabase1 ground truth average angle: ", degree_DB1/counter_DB1, " Total imgs: ", counter_DB1)
print ("MuensterDatabase ground truth avergage angle: ", degree_DB2/counter_DB2, " Total imgs: ", counter_DB2)
print ("zxingDatabase ground truth avergage angle: ", degree_DB3/counter_DB3, " Total imgs: ", counter_DB3)
print ("InsubriaDatabase2 ground truth avergage angle: ", degree_DB4/counter_DB4," Total imgs: ", counter_DB4)
print ("skkuinyougDatabase ground truth avergage angle: ", degree_DB5/counter_DB5, " Total imgs: ", counter_DB5)
print ("openfoodDatabase ground truth avergage angle: ", degree_DB6/counter_DB6, " Total imgs: ", counter_DB6)

