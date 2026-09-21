from setuptools import setup, find_packages

setup(
    name="LeTools-Learning",
    version="0.1",
    packages=find_packages(),
    package_data={"kuavo_deploy.msg": ["*.msg"]},
)
