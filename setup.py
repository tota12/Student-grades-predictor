from setuptools import setup, find_packages
from typing import List


def get_requirements(path: str) -> List[str]:
    '''Read requirements file and return list of requirements'''
    requirements = []
    with open(path, 'r') as file_obj:
        for line in file_obj:
            requirements.append(line.strip())
        if '-e .' in requirements:
            requirements.remove('-e .')
    return requirements


setup(
    name='endToEnd',
    version='0.0.1',
    author='tota',
    author_email='martina.mounir16@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
)
