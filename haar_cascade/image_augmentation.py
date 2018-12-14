import re
import os

import cv2
import imgaug as ia
from imgaug import augmenters as iaa


def update_infolist_with_originalimage(f_open):
    lines=f_open.readlines()    

    image_name=re.search(r'\d+', each_file).group()+ '.png' #lines[2][11:-12] #extract image name from second line

    # print image_name

    # print lines[19]

    x_min= int(re.search(r'\d+', lines[19]).group()) #xmin is in 19th line and similarly for the others.
    y_min= int(re.search(r'\d+', lines[20]).group())

    x_max= int(re.search(r'\d+', lines[21]).group())
    y_max= int(re.search(r'\d+', lines[22]).group())

    to_be_written=image_name+" 1 "+str(x_min)+" "+str(y_min)+" "+str(abs(x_max-x_min))+" "+ str(abs(y_max-y_min)) +'\n'
    info_list.write(to_be_written)

    return image_name,x_min,y_min,x_max,y_max


##################################main program ########################
ia.seed(1)

filecount=0
skipped_filecount=0

files=os.listdir('labelled_data_total')

info_list=open('info.lst','a')

for each_file in files:
    f_open=open('labelled_data_total/'+each_file,'r')


    image_name,x_min,y_min,x_max,y_max=update_infolist_with_originalimage(f_open)


    # bounding box is 347 5 1090 846
    image = cv2.imread('images/'+image_name,1)
    size_y,size_x,size_z=image.shape

    bbs = ia.BoundingBoxesOnImage([
        ia.BoundingBox(x1=x_min, y1=y_min, x2=x_max, y2=y_max),
    ], shape=image.shape)
 


    for iteration in range(5):
        seq = iaa.Sometimes(0.5,
        iaa.Affine(
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            scale=(0.5, 1.2),
            shear=(-16,16),
            rotate=(-45,45),
        ), # translate by 40/60px on x/y axis, and scale to 50-70%, affects BBs
        iaa.SomeOf(4,[
        iaa.Multiply((1.2, 1.5)), # change brightness, doesn't affect BBs
        iaa.GaussianBlur(sigma=2.0),
        
        iaa.AdditiveGaussianNoise(scale=0.2*255),
        iaa.Add(50, per_channel=True),
        iaa.Sharpen(alpha=0.5),
        iaa.AverageBlur(k=(2, 11)),
        iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 2.0)),
        iaa.PiecewiseAffine(scale=(0.01, 0.05))
        ],random_order=True))

        # Make our sequence deterministic.
        # We can now apply it to the image and then to the BBs and it will
        # lead to the same augmentations.
        # IMPORTANT: Call this once PER BATCH, otherwise you will always get the
        # exactly same augmentations for every batch!
        seq_det = seq.to_deterministic()



        # Augment BBs and images.
        # As we only have one image and list of BBs, we use
        # [image] and [bbs] to turn both into lists (batches) for the
        # functions and then [0] to reverse that. In a real experiment, your
        # variables would likely already be lists.
        image_aug = seq_det.augment_images([image])[0]
        bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]

        # print coordinates before/after augmentation (see below)
        # use .x1_int, .y_int, ... to get integer coordinates
        for i in range(len(bbs.bounding_boxes)):
            before = bbs.bounding_boxes[i]
            after = bbs_aug.bounding_boxes[i]
            print("BB %d: (%.4f, %.4f, %.4f, %.4f) -> (%.4f, %.4f, %.4f, %.4f)" % (
                i,
                before.x1, before.y1, before.x2, before.y2,
                after.x1, after.y1, after.x2, after.y2)
            )

        x_min=min(after.x1,after.x2)
        y_min=min(after.y1,after.y2)

        x_max=max(after.x1,after.x2)
        y_max=max(after.y1,after.y2)

        if(x_min<0 or y_min<0 or x_max>size_x or y_max>size_y):
            print "skipped"
            skipped_filecount+=1
            print "skipped filecount=",skipped_filecount

            continue

        to_be_written=image_name[:-4]+'_'+str(iteration+1)+'.png'+" 1 "+str(int(x_min))+" "+str(int(y_min))+" "+str(int(abs(x_max-x_min)))+" "+ str(int(abs(y_max-y_min))) +'\n'
        info_list.write(to_be_written)



        # image with BBs before/after augmentation (shown below)
        image_before = bbs.draw_on_image(image, thickness=2)
        image_after = bbs_aug.draw_on_image(image_aug, thickness=2, color=[0, 0, 255])

        # cv2.imshow('image_before',image_before)
        # cv2.imshow('image_after',image_after)

        cv2.imwrite('images/'+image_name[:-4]+'_'+str(iteration+1)+'.png',image_after)
        print 'Location is ---'+'images/'+image_name[:-4]+'_'+str(iteration+1)+'.png'
        filecount+=1

        print "filecount=",filecount

        # cv2.waitKey(0)  
info_list.close()
f_open.close()

print "--------------End--------------"
print "total iterations", filecount+skipped_filecount
print "skipped filecount=",skipped_filecount
print "filecount=",filecount