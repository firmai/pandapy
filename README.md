## Shipping pip packages — the automated way

[ITNEXT: Create, build and ship a Python3 pip module in 5 minutes](https://itnext.io/create-build-and-ship-a-python3-pip-module-in-5-minutes-31dd6d9d5c8f?source=friends_link&sk=311381bad115c64f022902e62aa582ff)

### Prerequisites
* PyPi account. Register [here](https://pypi.org/account/register/) and save your credentials in a safe place.
* Docker. Install instructions: [Ubuntu](https://docs.docker.com/install/linux/docker-ce/ubuntu/#install-docker-ce) or [Mac](https://docs.docker.com/docker-for-mac/install/).

You also need the ability to run make, have a POSIX compliant shell and a code editor — nearly every Linux-, Unix- or Mac system will work.

### PyShipper
PyShipper is an automation part, doing the grunt-work of setting up a module structure, adds some boilerplate code, and provides a pipeline to build, test and ship a module — saving time when you update or create a module.

The relevant code is mostly in Makefile, complemented with some Docker configuration and a slightly modified setup.py. For the curious readers, this previous [Docker & Makefile](https://itnext.io/docker-makefile-x-ops-sharing-infra-as-code-parts-ea6fa0d22946?source=friends_link&sk=1c42525c25039efadcbd25776a3019dd) article explains the base. PyShipper is a version that does the job of Python module delivery.

### Steps
 1. Think of a name
 2. Create a new module directory
 3. Configure variables
 4. Run and edit code
 5. Make module
 6. Publish

I will go through each step, and explain where and how PyShipper kicks in to automate a few things along the way.

### 1. Think of a name
If you find it difficult to come up with a good name, you are not alone:
>   There are only two hard things in Computer Science: cache invalidation and naming things.  — Phil Karlton

I always struggle with naming things. After hours in slow thinking mode, I usually end up with a name that describes what problem is solved — hence the name PyShipper. If a name is taken, pre- or postfixing the name with “Py”, “Script”, “Tool”,  or something similar usually works out. 

Naming is one of these things I have not yet figured out how to automate. If there are any machine learning related ideas, please share!

### 2. Create a new module structure
This [documentation](https://python-packaging.readthedocs.io/en/latest/minimal.html) on Python packages describes the required package structure. Even as we automated this step, it goes without saying the documentation is still essential reading material — useful for debugging.

Let’s go to the one-liners.

    # get a copy of PyShipper and change into it
    git clone [https://github.com/LINKIT-Group/pyshipper.git](https://github.com/LINKIT-Group/pyshipper.git)
    cd pyshipper

    # fork (copy/ paste) the contents to a new directory
    make fork dest=~/${YOUR_NEW_MODULE_NAME}

This forks a stripped version of PyShipper to a new directory called ~/${YOUR_NEW_MODULE_NAME}. Files like LICENSE and README.md are not copied. After all, you own the new module, and thus need to add a license and documentation to it — no license strings attached.

### 3. Configure variables
Our next step is to change to the new directory and edit the /variables file.

    # switch to the new module
    cd ~/${YOUR_NEW_MODULE_NAME}

    # edit the /variables file
    {replace_with_your_editor} variables

The NAME variable in /variables is picked up by Makefile, and all variables are exported to the environment of a Docker container,  used by setup.py during container execution.

### 4. Run and edit code
PyShipper ships with a /module directory containing boilerplate code for a module. There is also a coding pattern, with a minimal “hello-world”-like function, that can be called both CLI- and import-style.

Of course all Python3 — as Python2 goes [EOL Januari 2020](https://www.ncsc.gov.uk/blog-post/time-to-shed-python-2)!

    # enter the container runtime
    make shell

    # test run the module in CLI
    python3 -m module --name "PyShipper"

    # start Python3, import the module and test
    python3

    >>> import module
    >>> module.main(name="PyShipper")

This runs the code under /module. Instead of using the name “module”, you can also replace it with the name of your own module. A symlink is created to make both work — in the shipped version, only your own name can be used to reference the module.

If you are a Python Developer, I am sure you need to know what do next. All module code to edit is in the /module — have fun tinkering!

### 5. Make module
When you have a minimal working version it is time to build the package. One thing you may want to do if check the VERSION in /variables first and ensure it’s updated — I just made a note to myself to automate that in a next version, who likes keeping track of versions? ;).

    # build the module -- this runs setup.py in the container
    make module

    # better practice version of the former
    # includes pylint code quality testing
    make pylint module

### 6. Publish
The output of the build process is a gzipped tar archive, present in the /dist directory. By uploading this file to [PyPi](https://pypi.org/) —Python Package Index — the module is published, and install-able through Python pip by everyone.

    # upload the module
    make upload

The command above prompts for a username and password. You need to insert your [PyPi](https://pypi.org/) credentials—see Prerequisites at the beginning of this chapter.

After you created the module, published it on [PyPi](https://pypi.org/), you can install it on any capable system and use it like any other Python3 pip module.

    # install the module
    sudo python3 -m pip install ${YOUR_NEW_MODULE_NAME}

## Final thoughts
I already use this automation myself and I am quite happy with it. I regularly create small Python modules for specific tasks; it’s incredibly easy to import them in containers or server-less environments afterwards. 

Now with this piece of automation more repetitive work is cut — that means more time to spend on other innovations.

I hope you find this equally useful. I’d be delighted if this helps, at least to some of you, to contribute to the Python eco-system.

Happy Python module writing!
