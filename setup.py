from setuptools import find_packages,setup
from typing import List

def get_requirements(file_path:str)->List[str]:
    '''
    # Returns a list of requirements
    '''
    reqs = []
    with open(file_path) as file_obj:
        reqs = file_obj.readlines()
        reqs = [req.replace("\n","") for req in reqs]

        # Remove final character
        if "-e ." in reqs:
            reqs.remove("-e .")

setup(
    name = 'mlproject',
    version = "0.0.1",
    author = "SusheelP",
    author_email="susheelpatel.2022@gmail.com",
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt')
)