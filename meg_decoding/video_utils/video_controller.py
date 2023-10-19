import os
from typing import List, Optional

import cv2
import numpy as np
from torchvision.io import read_video
# import numpy.typing as npt


class VideoController(object):
    def __init__(self, movie_path: str, use_torchvision:bool=False):
        """Video Controller. this class has some useful method which get image from a video.

        Args:
            movie_path (str): movie file path

        Raises:
            FileNotFoundError: if file not found.
            ValueError: if file cannot be opened.
        """
        self.movie_path = movie_path
        # import pdb; pdb.set_trace()
        if self.movie_path != 0 and not os.path.exists(movie_path):
            print('while loading {}, following eception occurs.'.format(self.movie_path))
            raise FileNotFoundError("camera or video cannot be not opened.")
        # 複数processが同じ動画ファイルにアクセスするとデッドロックになる
        self.video_capture = cv2.VideoCapture(movie_path)

        if not self.video_capture.isOpened():
            print('while loading {}, following eception occurs.'.format(self.movie_path))
            raise ValueError("camera or video cannot be not opened.")

        self._frame_num = self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
        self._fps = self.video_capture.get(cv2.CAP_PROP_FPS)
        self.use_torchvision = use_torchvision
        if self.use_torchvision:
            print('Info: using torchvision read_video API')

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_num(self) -> int:
        return int(self._frame_num)

    @property
    def current_frame(self) -> float:
        return self.video_capture.get(cv2.CAP_PROP_POS_FRAMES)

    def get_current_image(self) :#-> npt.NDArray[np.uint8]:
        """get image at current index of video.

        Raises:
            ValueError: if image cannot be extracted.

        Returns:
            npt.NDArray[np.uint8]: image as numpy array (height, width. color).
        """
        ret, frame = self.video_capture.read()
        if frame is None:
            raise ValueError("cannot read frame.")
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # ADd by inoue 20230906
        return frame

    def get_image_at(self, frame_idx: int) :# -> npt.NDArray[np.uint8]:
        """get image at designated frame index.

        Args:
            frame_idx (int): frame index.

        Returns:
            npt.NDArray[np.uint8]: image as numpy array (height, width. color).
        """
        if self.use_torchvision:
            start_sec = frame_idx / self.fps # [frame] / [frame/sec]
            frame, _, _ = read_video(self.movie_path, start_pts=start_sec, end_pts=start_sec, pts_unit='sec', output_format='THWC')
            # print('Video Controler:', 'SUM:', frame.sum(), 'START: ', start_sec, self.movie_path )
            # import pdb; pdb.set_trace()
            return frame[0].numpy()
        else:
            if 0 <= frame_idx - self.current_frame <= 10:
                im = np.array([], dtype="uint8")
                for _ in range(int(frame_idx - self.current_frame + 1)):
                    im = self.get_current_image()
                return im

            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            return self.get_current_image()

    def get_random_image(self) :#-> npt.NDArray[np.uint8]:
        frame_idx = np.random.random_integers(0, int(self.frame_num) - 1)
        return self.get_image_at(frame_idx=frame_idx)

    def __del__(self):
        if self.video_capture is not None:
            self.video_capture.release()


class VideoNotInitialized(Exception):
    pass


class VideoSaver:
    def __init__(
        self, video_path: str, fps=30.0, video_format: Optional[List[str]] = None
    ) -> None:
        """VideoSaver. A wrapper class of cv2.VideoWriter

        Args:
            video_path (str): video path for saving.
            fps (float, optional): fps. Defaults to 30.0.
            video_format (list): video format. default is ["m", "p", "4", "v"]

        """
        self._video_path: str = video_path
        self._video: Optional[cv2.VideoWriter] = None
        self._fps = fps
        self._video_format = video_format or ["m", "p", "4", "v"]
        self._frame_num = 0
        self._img_size_h = -1
        self._img_size_w = -1

    @property
    def video_path(self) -> str:
        return str(self._video_path)

    @property
    def fps(self) -> float:
        return float(self._fps)

    @property
    def frame_num(self) -> int:
        return int(self._frame_num)

    @property
    def image_size(self) -> tuple:
        if self._img_size_h == -1 or self._img_size_w == -1:
            raise VideoNotInitialized("not initialized yet. put image first.")
        return (self._img_size_h, self._img_size_w)

    def write(self, image: np.ndarray) -> None:
        """put image into a video.

        Args:
            image (np.ndarray): Shape is h x w x c and channel format is BGR.

        Raises:
            ValueError: raise Error if image size changed.
        """
        if self._video is None:
            img_size_w = image.shape[1]
            img_size_h = image.shape[0]
            self._img_size_h = img_size_h
            self._img_size_w = img_size_w

            fourcc = cv2.VideoWriter_fourcc(*self._video_format)
            os.makedirs(os.path.dirname(self._video_path), exist_ok=True)
            self._video = cv2.VideoWriter(
                self._video_path, fourcc, self._fps, (img_size_w, img_size_h)
            )
        if image.shape[1] != self._img_size_w or image.shape[0] != self._img_size_h:
            raise ValueError("all image size must be same.")
        self._video.write(image)
        self._frame_num += 1

    def __call__(self, image: np.ndarray) -> None:
        """put image into a video.

        Args:
            image (np.ndarray): Shape is h x w x c and channel format is BGR.
        """
        self.write(image)

    def release(self) -> None:
        """release video."""
        if self._video is not None:
            self._video.release()

    def __del__(self) -> None:
        self.release()