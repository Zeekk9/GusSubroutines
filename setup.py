from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="GusSubroutines",
    version="1.0.0",
    author="Dr. Carlos Augusto Flores Meneses",
    author_email="augustoflores94@gmail.com.com",
    description="Package of utilities for image and phase processing developed by Dr. Augusto Meneses.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zeekk9/GusSubroutines",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    keywords="image-processing phase-unwrapping interferometry",
    project_urls={
        "Bug Tracker": "https://github.com/tuusuario/GusSubroutines/issues",
        "Documentation": "https://github.com/tuusuario/GusSubroutines/wiki",
        "Source Code": "https://github.com/tuusuario/GusSubroutines",
    },
)