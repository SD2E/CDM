README Last Updated by Hamed on 3/26/21

# CDM

### Installation
1. Clone this repository into the environment of your choice (directory, conda env, virtualenv, etc). Conda envs are recommended.
*** Ignore steps 2 and 3 if you want more flexibility to install versions of your own choice
2. Using command-line or terminal, Navigate inside the cdm directory. You should be at the same level of requirements.txt.
3. Run `pip3 install -r requirements.txt` .
4. Using command-line or terminal, navigate to the directory in which you cloned this repo (not inside the cdm directory itself). This
   should be 1 level higher than where you were in the previous step.
5. Run `pip3 install ./cdm` or `pip3 install -e cdm` .
This will install the `cdm` package and make it visible to all other repositories/projects
you have in the current environment. The `-e` option stands for "editable". This will install the package
in a way where any local changes to the package will automatically be reflected in your environment.
See [this link](https://stackoverflow.com/questions/41535915/python-pip-install-from-local-dir/41536128)
for more details.

**Note 1:** Do not do `pip3 install cdm`! Because that will install a different cdm package that exists on pypi.
Instead do `pip3 install ./cdm` so pip knows to look at for a directory.
- For editable mode, it doesn't matter so feel free to do `pip3 install -e cdm`.

**Note 2:** On TACC you might not be able to install this package unless you use `-e` **and/or** `--user`: e.g. `pip3 install -e cdm --user`.