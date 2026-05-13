from setuptools import setup, find_packages
import os

# Read README for long description
def read_file(filename):
    with open(os.path.join(os.path.dirname(__file__), filename), encoding='utf-8') as f:
        return f.read()

setup(
    name="deep-thought-rl",
    version="0.1.0",
    author="Deep Thought Contributors",
    description="Adaptive Sparse Cognitive Network for Reinforcement Learning",
    long_description=read_file("README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/hollowguy898/deepthough",
    packages=find_packages(exclude=["tests", "examples", "docs"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "numpy>=1.21.0",
        "gymnasium>=0.26.0",
        "tensorboard>=2.10.0",
        "tqdm>=4.64.0",
        "pyyaml>=6.0",
        "wandb>=0.13.0",
        "opencv-python>=4.6.0",
        "pillow>=9.2.0",
        "scipy>=1.9.0",
        "matplotlib>=3.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.971",
            "pre-commit>=2.20.0",
        ],
        "atari": [
            "gymnasium[atari]>=0.26.0",
            "ale-py>=0.8.0",
        ],
        "mujoco": [
            "gymnasium[mujoco]>=0.26.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "deep-thought=deep_thought.cli:main",
        ],
    },
)
