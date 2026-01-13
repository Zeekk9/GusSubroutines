# setup.py
from setuptools import setup, find_packages

setup(
    name="GusSubroutines",
    version="1.0.0",
    author="Tu Nombre",
    author_email="tu@email.com",
    description="Paquete de funciones personalizadas para procesamiento de interferometría",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/tuusuario/GusSubroutines",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=[
        'numpy>=1.21.0',
        'opencv-python>=4.5.0',
        'scipy>=1.7.0',
        'matplotlib>=3.4.0',
    ],
)