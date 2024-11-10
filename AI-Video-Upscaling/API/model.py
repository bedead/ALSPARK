import numpy as np
import os
import subprocess
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from os import path as osp
from tqdm import tqdm
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

try:
    import ffmpeg
except ImportError:
    subprocess.check_call(["pip", "install", "ffmpeg-python==0.2.0"])
    import ffmpeg

torch.backends.cudnn.benchmark = True


class Reader:
    def __init__(self, video_path):
        self.video_path = video_path
        (
            self.stream_reader,
            self.width,
            self.height,
            self.fps,
            self.audio,
            self.nb_frames,
        ) = self._initialize_video_stream()

    def _initialize_video_stream(self):
        probe = ffmpeg.probe(self.video_path)
        video_streams = [
            stream for stream in probe["streams"] if stream["codec_type"] == "video"
        ]
        has_audio = any(stream["codec_type"] == "audio" for stream in probe["streams"])
        width = video_streams[0]["width"]
        height = video_streams[0]["height"]
        fps = eval(video_streams[0]["avg_frame_rate"])
        audio = ffmpeg.input(self.video_path).audio if has_audio else None
        nb_frames = int(video_streams[0]["nb_frames"])
        stream_reader = (
            ffmpeg.input(self.video_path)
            .output("pipe:", format="rawvideo", pix_fmt="bgr24", loglevel="error")
            .run_async(pipe_stdin=True, pipe_stdout=True)
        )
        return stream_reader, width, height, fps, audio, nb_frames

    def get_resolution(self):
        return self.height, self.width

    def get_fps(self):
        return self.fps

    def get_audio(self):
        return self.audio

    def get_frame(self):
        img_bytes = self.stream_reader.stdout.read(self.width * self.height * 3)
        if not img_bytes:
            return None
        img = np.frombuffer(img_bytes, np.uint8).reshape([self.height, self.width, 3])
        return img

    def close(self):
        self.stream_reader.stdin.close()
        self.stream_reader.wait()


class Writer:
    def __init__(self, audio, height, width, video_save_path, fps, outscale):
        self.stream_writer = self._initialize_writer(
            audio, height, width, video_save_path, fps, outscale
        )

    def _initialize_writer(self, audio, height, width, video_save_path, fps, outscale):
        out_width, out_height = int(width * outscale), int(
            height * outscale
        )  # Update as per outscale
        if audio:
            return (
                ffmpeg.input(
                    "pipe:",
                    format="rawvideo",
                    pix_fmt="bgr24",
                    s=f"{out_width}x{out_height}",
                    framerate=fps,
                )
                .output(
                    audio,
                    video_save_path,
                    pix_fmt="yuv420p",
                    vcodec="libx264",
                    acodec="copy",
                    loglevel="error",
                )
                .overwrite_output()
                .run_async(pipe_stdin=True)
            )
        return (
            ffmpeg.input(
                "pipe:",
                format="rawvideo",
                pix_fmt="bgr24",
                s=f"{out_width}x{out_height}",
                framerate=fps,
            )
            .output(
                video_save_path, pix_fmt="yuv420p", vcodec="libx264", loglevel="error"
            )
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )

    def write_frame(self, frame):
        self.stream_writer.stdin.write(frame.astype(np.uint8).tobytes())

    def close(self):
        self.stream_writer.stdin.close()
        self.stream_writer.wait()


class RealESRGANProcessor:
    def __init__(
        self,
        model_name,
        device,
        outscale=2,
        denoise_strength=0.5,
        face_enhance=False,
        tile=0,
        tile_pad=5,
        pre_pad=0,
        workers=1,
    ):
        self.model_name = model_name
        self.device = device
        self.outscale = outscale
        self.denoise_strength = denoise_strength
        self.face_enhance = face_enhance
        self.tile = tile
        self.tile_pad = tile_pad
        self.pre_pad = pre_pad
        self.workers = workers  # Added workers input

        # Initialize the model
        self.model, self.netscale, self.file_url = self._get_model_details()
        self.upsampler = self._initialize_upsampler()

        if face_enhance:
            self.face_enhancer = self._initialize_face_enhancer()
        else:
            self.face_enhancer = None

    def _get_model_details(self):
        if self.model_name == "RealESRGAN_x4plus":
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=4,
            )
            netscale = 4
            file_url = [
                "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
            ]
        elif self.model_name == "realesr-animevideov3":
            model = SRVGGNetCompact(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_conv=16,
                upscale=4,
                act_type="prelu",
            )
            netscale = 4
            file_url = [
                "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth"
            ]
        elif self.model_name == "RealESRGAN_x2plus":  # x2 RRDBNet model
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=2,
            )
            netscale = 2
            file_url = [
                "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
            ]
        else:
            raise ValueError(f"Model '{self.model_name}' is not supported.")
        return model, netscale, file_url

    def _initialize_upsampler(self):
        model_path = self._download_model_weights()
        dni_weight = (
            [self.denoise_strength, 1 - self.denoise_strength]
            if self.model_name == "realesr-general-x4v3"
            else None
        )
        return RealESRGANer(
            scale=self.netscale,
            model_path=model_path,
            dni_weight=dni_weight,
            model=self.model,
            tile=self.tile,
            tile_pad=self.tile_pad,
            pre_pad=self.pre_pad,
            half=True if self.device == "cuda" else False,
            device=self.device,
        )

    def _download_model_weights(self):
        model_dir = osp.join("weights")
        os.makedirs(model_dir, exist_ok=True)
        for url in self.file_url:
            model_path = load_file_from_url(
                url=url, model_dir=model_dir, progress=True, file_name=None
            )
        return model_path

    def _initialize_face_enhancer(self):
        from gfpgan import GFPGANer

        return GFPGANer(
            model_path="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
            upscale=self.outscale,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=self.upsampler,
        )

    def process_video(self, video_path, save_path, fps=None):
        # reader = Reader(video_path)
        # print(f"Total frames : {reader.nb_frames}")
        # audio = reader.get_audio()
        # height, width = reader.get_resolution()
        # fps = fps or reader.get_fps()
        # writer = Writer(audio, height, width, save_path, fps, self.outscale)

        # frame_queue = deque()
        # pbar = tqdm(total=reader.nb_frames, unit='frame', desc='Processing video')

        # def process_frame(idx, img):
        #     if self.face_enhance:
        #         _, _, output = self.face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
        #     else:
        #         output, _ = self.upsampler.enhance(img, outscale=self.outscale)
        #     return idx , output

        # with ThreadPoolExecutor(max_workers=self.workers) as executor:
        #     futures = {executor.submit(process_frame, idx, reader.get_frame()): idx for idx in range(reader.nb_frames)}
        #     for future in as_completed(futures):
        #         idx, output = future.result()
        #         try:
        #             frame_queue.append((idx, output))  # Append frame in order of submission
        #             frame_queue = deque(sorted(frame_queue, key=lambda x: x[0]))
        #         except RuntimeError as error:
        #             print('Error:', error)
        #             print('Try reducing the tile size if CUDA out of memory.')

        #         # Write frames in order based on index
        #         while frame_queue and frame_queue[0][0] == idx:
        #             _, frame = frame_queue.popleft()
        #             writer.write_frame(frame)
        #             pbar.update(1)

        # reader.close()
        # writer.close()
        reader = Reader(video_path)
        print(f"Total frames : {reader.nb_frames}")
        audio = reader.get_audio()
        height, width = reader.get_resolution()
        fps = fps or reader.get_fps()
        writer = Writer(audio, height, width, save_path, fps, self.outscale)

        pbar = tqdm(total=reader.nb_frames, unit="frame", desc="Processing video")

        # Process each frame in a sequential manner without multithreading
        for idx in range(reader.nb_frames):
            img = reader.get_frame()
            try:
                if self.face_enhance:
                    _, _, output = self.face_enhancer.enhance(
                        img, has_aligned=False, only_center_face=False, paste_back=True
                    )
                else:
                    output, _ = self.upsampler.enhance(img, outscale=self.outscale)

                writer.write_frame(output)  # Write the processed frame directly
                pbar.update(1)
            except RuntimeError as error:
                print("Error:", error)
                print("Try reducing the tile size if CUDA out of memory.")

        reader.close()
        writer.close()
