"""Copyright 2019 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from os import path
from setuptools import setup, find_packages

package_dir = path.abspath(path.dirname(__file__))
with open(path.join(package_dir, 'README.md')) as f:
  long_description = f.read()

setup(
    name='bam-intp',
    version='0.1',
    description='Benchmarking attribution methods.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/google-research-datasets/bam',
    author='Google Inc.',
    author_email='opensource@google.com',
    license='Apache 2.0',
    packages=find_packages(),
    zip_safe=False,
    install_requires=[
        'numpy',
        'tensorflow',
        'Pillow>=6.0.0',
        'tcav>=0.2.1',
        'saliency>=0.0.2',
        'Cython>=0.29.12',
        'pycocotools>=2.0.0',
        'scikit-learn>=0.20.3',
    ])
