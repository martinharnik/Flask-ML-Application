from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT = "-e ."
def get_requirements(file_path:str) -> List[str]:
    '''
    Reads requirements from a file and returns them as a list
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = [req.strip() for req in file_obj.readlines()]
        
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
        
    return requirements
        

setup(
    name='ML-Project-01',
    version='0.1',
    author="Martin Harnik",
    author_email="martin.harnik91@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)