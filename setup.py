from setuptools import setup, find_packages

setup(
    name=\'BETA_pacote\',
    version=\'0.1.0\',
    packages=find_packages(),
    install_requires=[
        \\'lightkurve\\',
        \\'matplotlib\\',
        \\'numpy\\',
        \\'plotly\\',
        \\'lmfit\\',
        \\'scipy\\',
    ],
    author=\'Alessandro, Romualdo, Danilo\',
    description=\'Uma biblioteca para análise de curvas de luz do TESS.\',
    long_description=open(\'README.md\').read(),
    long_description_content_type=\'text/markdown\',
    url=\'https://github.com/yourusername/BETA_pacote\', # Substitua pelo seu repositório
    classifiers=[
        \\'Programming Language :: Python :: 3\\',
        \\'License :: OSI Approved :: MIT License\\',
        \\'Operating System :: OS Independent\\',
    ],
    python_requires=\'>=3.6\',
 )

