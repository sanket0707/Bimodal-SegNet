# Bimodal SegNet: Instance Segmentation Fusing Events and RGB Frames for Robotic Grasping

# Cite the article:


# Dataset repository:
https://kuacae-my.sharepoint.com/personal/100049863_ku_ac_ae/_layouts/15/onedrive.aspx?id=%2Fpersonal%2F100049863%5Fku%5Fac%5Fae%2FDocuments%2FESD&ga=1

# Dataset structure:

|-- ESD-1 (training)
    |-- conditions_1
        |-- RGB
            |-- images                   (raw RGB images)
            |-- masks                    (annotated masks)
        |-- events
            |-- left.mat                 (events info from left event camera, RGBD info, camera's movement info)
            |-- right.mat                (events info from right event camera, RGBD info, camera's movement info)
            |-- events_frame.mat         (synchronous left and right image frames transformed from RGBD coordinate)
            |-- mask_events_frame.mat    (synchronous left and right masks transformed from RGBD coordinate)
    |-- conditions_2
        |-- ...
    ...

|-- ESD-2 (testing)
    |-- conditions_1
        |-- RGB
            |-- images
            |-- masks
        |-- events
            |-- left.mat
            |-- right.mat
            |-- events_frame.mat
            |-- mask_events_frame.mat
    |-- conditions_2
        |-- ...
    ...
