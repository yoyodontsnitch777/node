from .nodes.save_load_pose import TSSavePoseDataAsPickle, TSLoadPoseDataPickle
from .nodes.openpose_smoother import KPSSmoothPoseDataAndRender
from .nodes.load_video_batch import LoadVideoBatchListFromDir
from .nodes.rename_files import RenameFilesInDir
from .nodes.color_match import TSColorMatchSequentialBias
from .nodes.preview_image_metadata import PreviewImageNoMetadata
from .nodes.video_combine_metadata import TSVideoCombineNoMetadata


NODE_CLASS_MAPPINGS = {
    "TSSavePoseDataAsPickle": TSSavePoseDataAsPickle,
    "TSLoadPoseDataPickle": TSLoadPoseDataPickle,
    "TSPoseDataSmoother": KPSSmoothPoseDataAndRender,
    "TSLoadVideoBatchListFromDir": LoadVideoBatchListFromDir,
    "TSRenameFilesInDir": RenameFilesInDir,
    "TSColorMatch": TSColorMatchSequentialBias,
    "TSPreviewImageNoMetadata": PreviewImageNoMetadata,
    "TSVideoCombineNoMetadata": TSVideoCombineNoMetadata,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TSSavePoseDataAsPickle": "Save Pose Data (PKL)",
    "TSLoadPoseDataPickle": "Load Pose Data (PKL)",
    "TSPoseDataSmoother": "Pose Data Smoother",
    "TSLoadVideoBatchListFromDir": "Load Video Batch List From Dir",
    "TSRenameFilesInDir": "Rename Files In Dir",
    "TSColorMatch": "Color Match",
    "TSPreviewImageNoMetadata": "Preview Image No Metadata",
    "TSVideoCombineNoMetadata": "Video Combine No Metadata",
}

WEB_DIRECTORY = "web"
