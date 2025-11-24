from setuptools import find_packages, setup

setup(
    name="labDynamics",
    version="0.0.1",
    packages=find_packages(include=["labDynamics", "labDynamics.*", "cusadi", "cusadi.*"]),
    author="David Zong",
    maintainer="David Zong",
    maintainer_email="david.zong@example.com",
    license="BSD-3",
    description="Fast and simple RL algorithms implemented in pytorch",
    python_requires=">=3.11",
    install_requires=["casadi"],
)
