from setuptools import setup

setup(name='cats_dogs',
      version='0.0.0',
      description='Cat vs dog image classifier',
      url='https://github.com/robertibatullin/cats_dogs',
      author='Robert Ibatullin',
      packages=['cats_dogs'],
      install_requires=[
          'tensorflow==2.2.0',
          'opencv-python==4.5.1.48',
          'numpy==1.19.2',
          'imutils==0.5.4',
          'python-magic==0.4.22',
          'flask==1.1.2'],
      zip_safe=False,
      include_package_data=True)
