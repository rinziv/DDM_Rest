# DDM_Rest
This Repository is the backend for the [Didactic Data Mining Tool](https://github.com/rinziv/DDM).

# Execute this project
## Installing
In order to install and the execute this repo, you have to install all your dependencies:
```
pip3 install -r requirements.txt
```
Then you can download a ddm module dependency and execute the project
```
mkdir ddm
cd ddm
git clone https://github.com/riccotti/DidacticDataMining
touch __init__.py
cd ..
python3 ddmrest.py
```

## Installing all dependencies in a virtualenv
First of all, if you haven't already installed `virtualenv`
```
pip install virtualenv
```
then, inside the project directory:
```
virtualenv -p `which python3` .
source bin/activate
pip install -r requirements.txt
python ddmrest.py
```
## Run Docker Container
In order to run the docker container you can build your image using the `Dockerfile` or just type
```bash
docker run -p 5000:5000 alessandro308/ddm_rest
```

# External dependencies
This project imports [ddm module](https://github.com/riccotti/DidacticDataMining). Before to execute this project, remember to download these files and insert them in a module called `ddm` inside the root project folder. 

# Known Errors
If you execute this project on MacOS, you can find this error:
```
**RuntimeError**: Python is not installed as a framework. The Mac OS X backend will not be able to function...
```
In order to fix it, follow these steps ([complete solution](https://stackoverflow.com/questions/21784641/installation-issue-with-matplotlib-python))
 - I assume you have installed the pip matplotlib, there is a directory in you root called ~/.matplotlib.
 - Create a file ~/.matplotlib/matplotlibrc there and add the following code: `backend: TkAgg`