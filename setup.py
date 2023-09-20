import setuptools

setuptools.setup(
    name="audio_slowfast",
    version="0.0.1",
    description="Slow-Fast Auditory Streams For Audio Recognition",
    long_description=open("README.md").read().strip(),
    long_description_content_type="text/markdown",
    author="Evangelos Kazakos",
    url=f"https://github.com/ClementSicard/auditory-slow-fast",
    packages=setuptools.find_packages(),
    install_requires=[
        "torch",
        "librosa",
        "h5py",
        "wandb",
        "fvcore",
        "simplejson",
        "psutil",
        "tensorboard",
    ],
    extras_require={
        "test": ["pytest", "pytest-cov"],
        "doc": ["sphinx-rtd-theme"],
    },
    license="Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.",
    keywords="auditory slowfast audio recognition",
)
