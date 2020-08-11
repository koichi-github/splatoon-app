import cv2
import os
import argparse


def main():

    video_path = './sample2.mp4'
    output_dir = './sample/'
    # source_dir = f'{output_path}source/'
    base_name = 'img'
    ext = 'png'

    num_file = f"{output_dir}num.txt"
    if os.path.exists(num_file):
        with open(num_file, "r") as f:
            last_num = int(f.read())
    else:
        last_num = 0

    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    # os.makedirs(os.path.dirname(source_dir), exist_ok=True)

    # save_all_frames(video_path, img_dir_path, base_name, ext=ext)

    cap = cv2.VideoCapture(video_path)
    count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    N = int(count/fps) + 1

    for i in range(N):
        save_frame_sec(cap, fps, i, output_dir, last_num)

    with open(num_file, "w") as f:
        f.write(str(last_num + N))

def save_all_frames(video_path, dir_path, basename, ext='jpg'):
    cap = cv2.VideoCapture(video_path)
    '''
    すべてのフレームを画像ファイルとして保存

    指定したディレクトリに<basename>_<連番>.<拡張子>というファイル名で保存する。
    '''

    if not cap.isOpened():
        return

    os.makedirs(dir_path, exist_ok=True)
    base_path = os.path.join(dir_path, basename)

    digit = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))

    n = 0

    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imwrite('{}_{}.{}'.format(base_path, str(n).zfill(digit), ext), frame)
            n += 1
        else:
            return


def save_frame_sec(cap, fps, sec, output_dir, last_num):
    '''
    時間（秒数）で指定して画像ファイルとして保存
    '''

    source_dir = f"{output_dir}source/images/"
    special_dir = f"{output_dir}special/images/"
    kill_dir = f"{output_dir}kill/images/"
    ally_dir = f"{output_dir}ally/images/"
    enemy_dir = f"{output_dir}enemy/images/"

    os.makedirs(os.path.dirname(source_dir), exist_ok=True)
    os.makedirs(os.path.dirname(special_dir), exist_ok=True)
    os.makedirs(os.path.dirname(kill_dir), exist_ok=True)
    os.makedirs(os.path.dirname(ally_dir), exist_ok=True)
    os.makedirs(os.path.dirname(enemy_dir), exist_ok=True)

    if not cap.isOpened():
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, round(fps * sec))

    ret, img = cap.read()
    special_img = img[10:161, 1110:1261]
    ally_img = img[4:95, 334:609]
    enemy_img = img[4:95, 680:955]
    kill_img = img[504:715, 476:807]

    if ret:
        cv2.imwrite(f"{source_dir}img{last_num+sec}.png", img)
        cv2.imwrite(f"{special_dir}img{last_num+sec}.png", special_img)
        cv2.imwrite(f"{ally_dir}img{last_num+sec}.png", ally_img)
        cv2.imwrite(f"{enemy_dir}img{last_num+sec}.png", enemy_img)
        cv2.imwrite(f"{kill_dir}img{last_num+sec}.png", kill_img)


if __name__ == "__main__":
    
    main()