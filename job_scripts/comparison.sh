python scripts/mesh_to_pa.py --input_mesh_file /media/aditya/OS/data/toys_4k/toys4k_obj_files/airplane/airplane_007/mesh.obj --save_dir ../outputs/fastmode/airplane_007 --profile_path ../outputs/fastmode/airplane_007/profile.prof --fastmode 

python scripts/mesh_to_pa.py --input_mesh_file /media/aditya/OS/data/toys_4k/toys4k_obj_files/airplane/airplane_007/mesh.obj --save_dir ../outputs/basic/airplane_007 --profile_path ../outputs/basic/airplane_007/profile.prof

# Rest without profile
# /media/aditya/OS/data/toys_4k/toys4k_obj_files/fastmode/dinosaur_053/mesh.obj
python scripts/mesh_to_pa.py --input_mesh_file /media/aditya/OS/data/toys_4k/toys4k_obj_files/fastmode/dinosaur_053/mesh.obj --save_dir ../outputs/fastmode/dinosaur_053 --fastmode 
python scripts/mesh_to_pa.py --input_mesh_file /media/aditya/OS/data/toys_4k/toys4k_obj_files/basic/dinosaur_053/mesh.obj --save_dir ../outputs/basic/dinosaur_053 


# shark
python scripts/mesh_to_pa.py --input_mesh_file /media/aditya/OS/data/toys_4k/toys4k_obj_files/shark/shark_014/mesh.obj --save_dir ../outputs/fastmode/shark_014 --profile_path ../outputs/fastmode/shark_014/profile.prof --fastmode
python scripts/mesh_to_pa.py --input_mesh_file /media/aditya/OS/data/toys_4k/toys4k_obj_files/shark/shark_014/mesh.obj --save_dir ../outputs/basic/shark_014 --profile_path ../outputs/basic/shark_014/profile.prof

# Rest without profile - Fastmode first
python scripts/mesh_to_pa.py --input_mesh_file /media/aditya/OS/data/toys_4k/toys4k_obj_files/monkey/monkey_002/mesh.obj --save_dir ../outputs/fastmode/monkey_002 --fastmode
python scripts/mesh_to_pa.py --input_mesh_file /media/aditya/OS/data/toys_4k/toys4k_obj_files/cake/cake_001/mesh.obj --save_dir ../outputs/fastmode/cake_001 --fastmode
python scripts/mesh_to_pa.py --input_mesh_file /media/aditya/OS/data/toys_4k/toys4k_obj_files/fox/fox_039/mesh.obj --save_dir ../outputs/fastmode/fox_039 --fastmode
python scripts/mesh_to_pa.py --input_mesh_file /media/aditya/OS/data/toys_4k/toys4k_obj_files/elephant/elephant_018/mesh.obj --save_dir ../outputs/fastmode/elephant_018 --fastmode
python scripts/mesh_to_pa.py --input_mesh_file /media/aditya/OS/data/toys_4k/toys4k_obj_files/penguin/penguin_023/mesh.obj --save_dir ../outputs/fastmode/penguin_023 --fastmode
python scripts/mesh_to_pa.py --input_mesh_file /media/aditya/OS/data/toys_4k/toys4k_obj_files/robot/robot_013/mesh.obj --save_dir ../outputs/fastmode/robot_013 --fastmode
python scripts/mesh_to_pa.py --input_mesh_file /media/aditya/OS/data/toys_4k/toys4k_obj_files/piano/piano_025/mesh.obj --save_dir ../outputs/fastmode/piano_025 --fastmode

# Basic mode (slow)
python scripts/mesh_to_pa.py --input_mesh_file /media/aditya/OS/data/toys_4k/toys4k_obj_files/monkey/monkey_002/mesh.obj --save_dir ../outputs/basic/monkey_002
python scripts/mesh_to_pa.py --input_mesh_file /media/aditya/OS/data/toys_4k/toys4k_obj_files/cake/cake_001/mesh.obj --save_dir ../outputs/basic/cake_001
python scripts/mesh_to_pa.py --input_mesh_file /media/aditya/OS/data/toys_4k/toys4k_obj_files/fox/fox_039/mesh.obj --save_dir ../outputs/basic/fox_039
python scripts/mesh_to_pa.py --input_mesh_file /media/aditya/OS/data/toys_4k/toys4k_obj_files/elephant/elephant_018/mesh.obj --save_dir ../outputs/basic/elephant_018
python scripts/mesh_to_pa.py --input_mesh_file /media/aditya/OS/data/toys_4k/toys4k_obj_files/penguin/penguin_023/mesh.obj --save_dir ../outputs/basic/penguin_023
python scripts/mesh_to_pa.py --input_mesh_file /media/aditya/OS/data/toys_4k/toys4k_obj_files/robot/robot_013/mesh.obj --save_dir ../outputs/basic/robot_013
python scripts/mesh_to_pa.py --input_mesh_file /media/aditya/OS/data/toys_4k/toys4k_obj_files/piano/piano_025/mesh.obj --save_dir ../outputs/basic/piano_025
