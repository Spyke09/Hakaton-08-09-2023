from setuptools import setup, find_packages

setup(name='myvoicehakaton2023',
      version='0.2',
      url='https://github.com/Spyke09/Hakaton-08-09-2023',
      license='MIT',
      author='Spyke09',
      description='Text clusterer for hackathon 09/08/2023',
      packages=find_packages(exclude=['tests']),
      long_description=open('README.md').read(),
      zip_safe=False)
