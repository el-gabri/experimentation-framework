from setuptools import setup, find_packages

setup(
    name="supply_experiments",
    version="1.0.2",
    description="Framework de experimentação geográfica de supply — iFood Groceries",
    packages=find_packages(exclude=["tests", "notebooks"]),
    python_requires=">=3.9",
    install_requires=["numpy>=1.23", "pandas>=1.5", "scipy>=1.9"],
)
