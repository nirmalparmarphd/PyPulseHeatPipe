import setuptools

with open("README.md", "r", encoding = "utf-8") as fh:
    long_description = fh.read()

setuptools.setup(name='PyPulseHeatPipe',
        version='1.1.13',
        license='GNU',
        url='https://github.com/nirmalparmarphd/PyPulseHeatPipe',
        description='The data analysis Python package for the Pulsating Heat Pipe experimental data', 
        long_description=long_description,
        long_description_content_type='text/markdown',
        author='Nirmal Parmar, PhD',
        author_email='nirmalparmarphd@gmail.com',
        packages=setuptools.find_packages(),
        include_package_data=True,
        install_requires=['scikit-learn','numpy', 'gitpython', 'pandas','scipy', 'matplotlib', 'seaborn', 'openpyxl', 'pygwalker', 'streamlit'],
        zip_safe=False)