# yowie-benchmarks
Benchmarks for Planning, Optimal Control and Reinforcement Learning


## Installation


Invoke ```pip``` at the root of the folder where you cloned this repo

```
$ pip install --user -e .
```


## Domains


### RDDL

#### Linear Quadratic Regulator, 1 dimensional

```
python3 -m yowie_rddl.lqr1d --seed <value> --num-instances <number of instances to generate>
```

#### Linear Quadratic Gaussian problem, 1 dimensional

```
python3 -m yowie_rddl.lqg1d --seed <value> --num-instances <number of instances to generate>
```


#### Linear Quadratic Gaussian problem, 2 dimensional, multiple units

```
python3 -m yowie_rddl.lqg2dmu --seed <value> --num-instances <number of instances to generate>
```
