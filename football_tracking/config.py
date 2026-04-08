from dataclasses import dataclass

@dataclass
class PipelineConfig:
    project_root: str
    input_dir: str
    output_dir: str
    roi_dir: str
    model_path: str
    processed_manifest_path: str
    fps: int = 25
    seconds_before: int = 10
    seconds_after: int = 10
    conf_threshold: float = 0.25
    environment: str = "local"   # "local" or "colab"