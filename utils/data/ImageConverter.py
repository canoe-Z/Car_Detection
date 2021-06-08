import cv2


class ImageConverter():
    def images2video(self, dst: str, images, fps: int):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        videoWriter = cv2.VideoWriter(
            dst, fourcc, fps, (960, 540), True)  # 最后一个是保存图片的尺寸
        for frame in images:
            videoWriter.write(frame)
        videoWriter.release()
