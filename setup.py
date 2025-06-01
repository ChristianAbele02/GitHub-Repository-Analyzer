"""
GitHub Repository Analyzer Setup Script

This setup script allows for easy installation of the GitHub Repository Analyzer
package and its dependencies. It supports both development and production installations.
"""
import os
from pathlib import Path
from setuptools import setup, find_packages


# Read the README file for the long description
def read_readme():
    try:
        with open("README.md", "r", encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        return "GitHub Repository Analyzer - A comprehensive tool for analyzing GitHub repositories"


# Read requirements from requirements.txt
def read_requirements():
    try:
        with open("requirements.txt", "r", encoding="utf-8") as fh:
            return [line.strip() for line in fh if line.strip() and not line.startswith("#")]
    except FileNotFoundError:
        return []

# Version management
def get_version():
    version_file = Path("VERSION")
    if version_file.exists():
        return version_file.read_text().strip()
    return "1.0.0"


# Package metadata
setup(
    name="github-repository-analyzer",
    version=get_version(),
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive tool for analyzing GitHub repositories and predicting development trends",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/github-repository-analyzer",
    license="MIT",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/github-repository-analyzer/issues",
        "Documentation": "https://github.com/yourusername/github-repository-analyzer/wiki",
        "Source Code": "https://github.com/yourusername/github-repository-analyzer",
    },

    # Package configuration
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Internet :: WWW/HTTP :: Dynamic Content",
    ],

    # Python version requirement
    python_requires=">=3.7",

    # Install dependencies
    install_requires=read_requirements(),

    # Optional dependencies for development and testing
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.10",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
            "pre-commit>=2.10",
        ],
        "docs": [
            "sphinx>=3.0",
            "sphinx-rtd-theme>=0.5",
            "myst-parser>=0.15",
        ],
        "jupyter": [
            "jupyter>=1.0",
            "ipykernel>=5.0",
            "ipywidgets>=7.5",
        ],
    },

    # Entry points for command-line interface
    entry_points={
        "console_scripts": [
            "github-analyzer=github_analyzer.main:main",
        ],
    },

    # Include additional files
    include_package_data=True,
    package_data={
        "github_analyzer": [
            "config/*.yaml",
            "templates/*.html",
        ],
    },

    # Keywords for package discovery
    keywords=[
        "github", "repository", "analysis", "trends", "data-science",
        "machine-learning", "visualization", "api", "statistics"
    ],

    # Zip safe configuration
    zip_safe=False,
)


# Post-installation setup and instructions
def post_install_message():
    print("""

    GitHub Repository Analyzer installed successfully!

    Next Steps:

    1. Set up your GitHub token:
       export GITHUB_TOKEN="your_personal_access_token"

    2. Or create a .env file:
       echo "GITHUB_TOKEN=your_token_here" > .env

    3. Test your installation:
       python -c "from main import GitHubRepositoryAnalyzer; print('Installation successful!')"
    Happy analyzing!
    """)
def validate_setup_requirements():
    required_files = ["README.md", "requirements.txt", "LICENSE"]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print(f"Warning: Missing required files: {', '.join(missing_files)}")
        return False
    return True

if __name__ == "__main__":
    post_install_message()
