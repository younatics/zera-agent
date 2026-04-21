from setuptools import find_packages, setup


INSTALL_REQUIRES = [
    "streamlit",
    "pandas",
    "plotly",
    "httpx",
    "watchdog",
    "python-dotenv",
    "openai",
    "anthropic",
    "requests",
    "datasets",
    "numpy",
    "rouge",
]


setup(
    name="prompt-autotuning-agent",
    version="0.1",
    packages=find_packages(
        include=[
            "agent",
            "agent.*",
            "evaluation",
            "evaluation.*",
            "scripts",
            "scripts.*",
        ]
    ),
    include_package_data=True,
    package_data={"agent": ["prompts/*.txt"]},
    install_requires=INSTALL_REQUIRES,
    extras_require={"dev": ["pytest>=7.0"]},
    python_requires=">=3.8",
)
