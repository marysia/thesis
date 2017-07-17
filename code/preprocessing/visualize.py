import os
import cv2

def array_to_mp4(array, filename, video_folder):
    """
    Function to visualise the lung nodule patch by creating a .mp4.

    Args:
        array: np.array, input array, presumably 13x120x120
        filename: str, prefix for .mp4
        video_folder: str, folder where the videos will be stored

    Note: function currently not in use due to cv2 install
    """
    video_file = os.path.join(video_folder, '%s.mp4' % filename)
    if not os.path.exists(video_folder):
        os.makedirs(video_folder)
    video = cv2.VideoWriter(video_file, fourcc=0x21, fps=5, frameSize=(array.shape[2], array.shape[1]), isColor=True)

    if video.isOpened():
        for frame_nr in range(array.shape[0]):
            frame = cv2.cvtColor(array[frame_nr], cv2.COLOR_GRAY2BGR)
            video.write(frame)
        video.release()
    else:
        print('An error occurred. Invalid codec?')

def array_to_png_folder(array, folder_name):
    """
    Saves each slice as a png in a folder.

    Args:
        array: np.array of lung nodule
        folder_name: str, where the pngs will be saved.
    """
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    for i in xrange(array.shape[0]):
        img = array[i]
        fname = str(i) + '.png' if i > 9 else '0' + str(i) + '.png'
        fname = os.path.join(folder_name, fname)
        cv2.imwrite(fname, img)