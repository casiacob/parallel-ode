Clone the repository
```
$ git clone https://github.com/casiacob/parallel-ode.git
```

Create a conda environment and install the package
```
$ conda create --name par-ode python=3.10
$ conda activate par-ode
$ cd parallel-ode
$ pip install .
```

Run an example (requires GPU).
```
$ cd examples
$ python logistic_runtime.py
```
