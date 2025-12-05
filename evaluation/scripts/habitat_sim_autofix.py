import site
import glob
import os
import shutil

replacement_file = os.path.join(os.getcwd(), "evaluation", "replace", "common.py")

site_pkgs = site.getsitepackages()

habitat_sim_paths = [
    path for pkg in site_pkgs for path in glob.glob(os.path.join(pkg, 'habitat_sim-0.1.7*.egg', 'habitat_sim', 'utils', 'common.py'))
]

if habitat_sim_paths:
    file_path = habitat_sim_paths[0]

    if os.path.exists(replacement_file):
        shutil.copy2(replacement_file, file_path)
        print(f'Replaced {file_path} with {replacement_file}')
    else:
        print(f'Error: Replacement file {replacement_file} not found!')
else:
    print('Error: habitat_sim/utils/common.py not found!')
