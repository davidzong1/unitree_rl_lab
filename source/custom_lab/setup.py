from setuptools import find_packages, setup

setup(
    name="custom_lab",
    version="0.0.1",
    packages=find_packages(include=["custom_lab", "custom_lab.*", "rsl_rl", "rsl_rl.*"]),
    author="David Zong",
    maintainer="David Zong",
    maintainer_email="david.zong@example.com",
    license="BSD-3",
    description="Fast and simple RL algorithms implemented in pytorch",
    python_requires=">=3.11",
    install_requires=[
        "isaacsim>=5.1.0.0",
        "isaaclab>=2.3.0",
    ],
)
