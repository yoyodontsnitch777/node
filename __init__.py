from .nodes.save_load_pose import TSSavePoseDataAsPickle, TSLoadPoseDataPickle
from .nodes.openpose_smoother import KPSSmoothPoseDataAndRender
from .nodes.load_video_batch import LoadVideoBatchListFromDir
from .nodes.rename_files import RenameFilesInDir
from .nodes.color_match import TSColorMatchSequentialBias
from .nodes.preview_image_metadata import PreviewImageNoMetadata
from .nodes.video_combine_metadata import TSVideoCombineNoMetadata


NODE_CLASS_MAPPINGS = {
    "CloserToolsSavePoseData": TSSavePoseDataAsPickle,
    "CloserToolsLoadPoseData": TSLoadPoseDataPickle,
    "CloserToolsPoseSmoother": KPSSmoothPoseDataAndRender,
    "CloserToolsLoadVideoBatch": LoadVideoBatchListFromDir,
    "CloserToolsRenameFiles": RenameFilesInDir,
    "CloserToolsColorMatch": TSColorMatchSequentialBias,
    "CloserToolsPreviewImage": PreviewImageNoMetadata,
    "CloserToolsVideoCombine": TSVideoCombineNoMetadata,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CloserToolsSavePoseData": "CloserTools Save Pose Data (PKL)",
    "CloserToolsLoadPoseData": "CloserTools Load Pose Data (PKL)",
    "CloserToolsPoseSmoother": "CloserTools Pose Data Smoother",
    "CloserToolsLoadVideoBatch": "CloserTools Load Video Batch List From Dir",
    "CloserToolsRenameFiles": "CloserTools Rename Files In Dir",
    "CloserToolsColorMatch": "CloserTools Color Match",
    "CloserToolsPreviewImage": "CloserTools Preview Image No Metadata",
    "CloserToolsVideoCombine": "CloserTools Video Combine No Metadata",
}

WEB_DIRECTORY = "web"
