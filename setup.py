"""Setup configuration for PrivacyFlow."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="privacyflow",
    version="0.1.0",
    author="PrivacyFlow Team",
    author_email="privacy@example.com",
    description="Privacy-Preserving Person Detection System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/username/privacyflow",
    packages=find_packages(exclude=["tests", "tests.*", "scripts"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Security",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21.0",
        "opencv-python>=4.5.0",
        "onnxruntime>=1.10.0",
        "scikit-learn>=1.0.0",
        "schedule>=1.1.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
        ],
        "gpu": [
            "onnxruntime-gpu>=1.10.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "privacyflow=privacyflow.cli:main",
        ],
    },
)
