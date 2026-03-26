USE_CUDA = True
CLEAN_UP_DELTA = 0.005
MIN_VOLUME_LIMIT = 0.0001

# Explicit base paths (edit for your machine/cluster).
DATA_BASE = "/users/aganesh8/data/aganesh8/data"
PROJECT_BASE = "/users/aganesh8/data/aganesh8/projects/project_neo"
OUTPUTS_BASE = "/oscar/data/dritchi1/aganesh8/data/project_neo/outputs"

# Derived paths.
TOY4K_PATH_PREFIX = f"{DATA_BASE}/toys4k_obj_files"
PARTOBJAVERSE_BASE = f"{DATA_BASE}/partobjaverse"
PARTOBJAVERSE_MESH_DIR = f"{PARTOBJAVERSE_BASE}/PartObjaverse-Tiny/PartObjaverse-Tiny_mesh"
PARTOBJAVERSE_INSTANCE_DIR = f"{PARTOBJAVERSE_BASE}/PartObjaverse-Tiny_instance_gt"
AOT_ARTIFACT_DIR = f"{DATA_BASE}/project_neo/aot"
SAVE_DIR_BASE = OUTPUTS_BASE
TOY4K_CSV_FILE = "superfit/dataset/spilts/new_testset.csv"
SEMANTIC_LOC = f"{PROJECT_BASE}/PartField/"