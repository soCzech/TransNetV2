from setuptools import setup

setup(
    name="transnetv2",
    version="1.0.0",
    # let user install tensorflow, etc. manually
    # install_requires=[
    #     "tensorflow>=2.0",
    #     "ffmpeg-python",
    #     "pillow"
    # ],
    entry_points={
        "console_scripts": [
            "transnetv2_predict = transnetv2.transnetv2:main",
        ]
    },
    packages=["transnetv2"],
    package_dir={"transnetv2": "./inference"},
    package_data={"transnetv2": [
        "transnetv2-weights/*",
        "transnetv2-weights/variables/*"
    ]},
    zip_safe=False
)
