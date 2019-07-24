from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name             = 'image_structure',
      version          = '0.1',
      author           = 'Anthony M. DeGennaro',
      author_email     = 'adegennaro@bnl.gov',
      description      = 'Python tools for structure function computation/analysis on 2d/3d images',
      long_description = readme(),
      classifiers      = [
        'Topic :: Image Postprocessing',
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: ISC License',
        'Programming Language :: Python :: 3.6',
      ],
      keywords         = 'experimental design image structure',
      url              = 'http://github.com/adegenna/image_structure',
      license          = 'ISC',
      packages         = ['image_structure','image_structure.src','image_structure.scripts'],
      package_dir      = {'image_structure'         : 'image_structure' , \
                          'image_structure.src'     : 'image_structure/src' ,\
                          'image_structure.scripts' : 'image_structure/scripts'},
      entry_points     = { 'console_scripts': ['Package = image_structure.scripts.driver:main' ] },
      install_requires = [ 'numpy', 'scipy', 'matplotlib' , 'lmfit' , 'scikit-image' ],
      python_requires  = '>=3',
      zip_safe         = False
)
