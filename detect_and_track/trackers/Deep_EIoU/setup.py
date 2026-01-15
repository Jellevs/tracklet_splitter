from setuptools import setup, find_packages

setup(
    name="deep-eiou-tracker",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "opencv-python",
        "supervision",
        "torch",
        "torchvision",
        "lap",
        "cython-bbox",
    ],
    author="Your Name",
    description="Simple Deep-EIoU tracker interface",
    python_requires=">=3.8",
)