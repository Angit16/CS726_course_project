MSRA Hand Tracking database is described in the paper
Realtime and Robust Hand Tracking from Depth, Chen Qian, Xiao Sun, Yichen Wei, Xiaoou Tang and Jian Sun, CVPR 2014.
Please cite the paper if you use this database.

In total 6 subjects' right hands are captured using Intel's Creative Interactive Gesture Camera. Each subject is asked to make various rapid gestures in a 400-frame video sequence. To account for different hand sizes, a global hand model scale is specified for each subject: 1.1, 1.0, 0.9, 0.95, 1.1, 1.0 for subject 1~6, respectively.

The camera intrinsic parameters are: principle point = image center(160, 120), focal length = 241.42.

The depth image is 320x240, each *.bin file stores the depth pixel values in row scanning order, which are 320*240 floats. The unit is millimeters. The bin file is binary and needs to be opened with std::ios::binary flag.

joint.txt file stores 400 frames x 21 hand joints per frame. Each line has 3 * 21 = 63 floats for 21 3D points in (x, y, z) coordinates. The 21 hand joints are: wrist, index_mcp, index_pip, index_dip, index_tip, middle_mcp, middle_pip, middle_dip, middle_tip, ring_mcp, ring_pip, ring_dip, ring_tip, little_mcp, little_pip, little_dip, little_tip, thumb_mcp, thumb_pip, thumb_dip, thumb_tip.

The corresponding *.jpg file is just for visualization of depth and ground truth joints.

For any questiones, please send email to xias@microsoft.com