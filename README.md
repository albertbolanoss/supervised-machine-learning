# Supervise Machine Learning

## Requirements

- [pyenv](https://pypi.org/project/pyenv/)
- [Poetry](https://python-poetry.org/docs/)

## Install Pyenv and Poetry

```sh
# Using pipx
python3 -m pip install --user pipx
pipx install pyenv
pipx install poetry

# Download Poetry
curl -sSL https://install.python-poetry.org | python3 -
export PATH="$HOME/.local/bin:$PATH" # Add the following line to any of these files ~/.bashrc, ~/.zshrc o ~/.bash_profile
source ~/.bashrc  # o source ~/.zshrc
```

## Create Project

```sh
pyenv install 3.13.0                                                    # Install the required python version
pyenv local 3.13.0                                                      # Select this python versio to use in local mode
poetry init                                                             # Create the toml file, init the project
poetry config virtualenvs.in-project true                               # Create the virtual environment in `.venv/`
poetry add numpy pandas matplotlib scikit-learn notebook jupyter        # Add initial dependencies
```

## Clone repository and install dependencies

```sh
git clone https://github.com/albertbolanoss/supervised-machine-learning.git
cd supervised-machine-learning.git
pyenv local 3.13.0
poetry install
```

## Other commands

```sh
jupyter notebook                                                        # Run Jupiter notebook server
```

## References:

- The theoretical content of this repository was taken from the book [Grokking Machine Learning](https://learning.oreilly.com/library/view/grokking-machine-learning/) written by Luis Serrano.
