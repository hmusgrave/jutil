import setuptools

def long_description():
    with open('README.md', 'r') as file:
        return file.read()

setuptools.setup(
    name='jutil',
    version='0.0.1',
    author='Hans Musgrave',
    author_email='Hans.Musgrave@gmail.com',
    description='A client-side rate limiter to avoid remote penalties',
    long_description=long_description(),
    long_description_content_type='text/markdown',
    url='https://github.com/hmusgrave/jutil',
    packages = ['jutil'],
    install_requires=[
        'jax',
        'jaxlib',
    ],
    python_requires='>=3.6',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)'
        'Operating System :: OS Independent',
    ],
)
