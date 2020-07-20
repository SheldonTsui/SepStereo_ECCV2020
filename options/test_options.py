from .base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)

        self.parser.add_argument('--input_audio_path', help='path to the input audio file')
        self.parser.add_argument('--video_frame_path', help='path to the input video frames')
        self.parser.add_argument('--output_dir_root', type=str, default='test_output', help='path to the output files')
        self.parser.add_argument('--input_audio_length', type=float, default=10, help='length of the testing video/audio')
        self.parser.add_argument('--hop_size', default=0.05, type=float, help='the hop length to perform audio spatialization in a sliding window approach')

        #model arguments
        self.parser.add_argument('--weights_visual', type=str, default='', help="weights for visual stream")
        self.parser.add_argument('--weights_audio', type=str, default='', help="weights for audio stream")
        self.parser.add_argument('--weights_fusion', type=str, default='', help="weights for fusion stream")
        self.parser.add_argument('--unet_ngf', type=int, default=64, help="unet base channel dimension")
        self.parser.add_argument('--unet_input_nc', type=int, default=2, help="input spectrogram number of channels")
        self.parser.add_argument('--unet_output_nc', type=int, default=2, help="output spectrogram number of channels")
        self.parser.add_argument('--use_fusion_pred', action='store_true', help='whether use fusion prediction for inference')

        self.mode = "test"
        self.isTrain = False
