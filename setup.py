import versioneer
try:
	from setuptools import setup, find_packages, Extension
	from setuptools.command.install import install
except ImportError:
	from distutils.core import setup, find_packages, Extension
	from distutils.command.install import install



with open('README.md', 'r') as f:
	long_description = f.read()

setup(
	name="brilleu",
	version=versioneer.get_version(),
	cmdclass=versioneer.get_cmdclass(),
	author="Greg Tucker",
	author_email="greg.tucker@stfc.ac.uk",
	description='an interface between brille and Euphonic',
	long_description=long_description,
	long_description_content_type="text/markdown",
	url="https://github.com/brille/brilleu",
	packages=find_packages(),
	classifiers=[
		"Development Status :: 3 - Alpha",
		"Programming Language :: Python :: 3",
		"License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
		"Operating System :: OS Independent",
		"Topic :: Scientific/Engineering"
	],
	python_requires='~=3.6',
	install_requires=[
		'brille==0.5.1',
		'euphonic==0.3.0',
		'pyyaml>=5',
		'requests'
	]
)
