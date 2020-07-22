README Last Updated by Hamed on 7/22/20

# CDM

### Installation
1. Clone this repository into the environment of your choice (directory, conda env, virtualenv, etc). Conda envs are recommended.
2. Using command-line, navigate to the directory in which you cloned this repo (not inside the repo itself).
3. Run `pip3 install cdm` or `pip3 install -e cdm` .
This will install the `cdm` package and make it visible to all other repositories/projects
you have in the current environment. The `-e` option stands for "editable". This will install the package
in a way where any local changes to the package will automatically be reflected in your environment.
See [this link](https://stackoverflow.com/questions/41535915/python-pip-install-from-local-dir/41536128)
for more details.

**Note 1**: I get an error when trying to install in editable mode with `-e`. For now only use `pip3 install cdm`.

**Note 2**: On Maverick2 you might not be able to install this package unless you use `--user`: e.g. `pip3 install -e cdm --user`