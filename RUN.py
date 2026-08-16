from config.config import Config
from scripts.run_rd4ad import run

OVERRIDES = dict(
    DATA_ROOT="dataset root path (contains good/ and defect/ subfolders)",
    GOOD_DIRNAME  = "good",
    DEFECT_DIRNAME = "defect",
    SPLIT_CACHE_PATH="splits/split_assignment.csv",
    EXPERIMENT="RD4AD_group1_wide_resnet50_2",
    BACKBONE="wide_resnet50_2",
    EPOCHS=100,
    THRESHOLD_PERCENTILE=95.0,
)

if __name__ == "__main__":
    cfg = Config(**OVERRIDES)
    run(cfg)
