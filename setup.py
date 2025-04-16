from setuptools import setup, find_packages

setup(
    name="prompt-autotuning-agent",
    version="0.1",
    packages=find_packages(include=['common', 'agent', 'dataset']),
    package_dir={
        'common': 'common',
        'agent': 'agent',
        'dataset': 'dataset'
    },
    install_requires=[
        "streamlit",
        "pandas",
        "plotly",
        "python-dotenv",
        "openai",
        "anthropic",
        "datasets",
        "numpy"
    ],
) 