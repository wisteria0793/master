from setuptools import setup, find_packages

setup(
    name="urban_analysis",
    version="0.1.0",
    description="Urban analysis tools for master thesis",
    author="Atsuya Katougi",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "torch",
        "tqdm",
        "requests",
        "python-dotenv",
        # "osmnx",  # 必要に応じて追加
        # "torch_geometric", # 必要に応じて追加
    ],
    entry_points={
        "console_scripts": [
            # 必要であればCLIコマンドを定義可能
            # "fetch-sv=collect_data.intelligent_street_view_fetcher:main",
        ],
    },
)
