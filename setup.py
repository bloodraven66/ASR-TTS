import setuptools



setuptools.setup(
    name='ASR_TTS',
    version='0.0.3',
    author='Mike Huls',
    author_email='mike_huls@hotmail.com',
    description='Testing installation of Package',
    url='https://github.com/bloodraven66/ASR_TTS',
    license='MIT',
    packages=['ASR_TTS'],
    install_requires=['ffmpeg-python',
                    'soundfile'],
)
