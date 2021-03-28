# gervigreind-project2

Project 2 for Artificial Intelligence course @ Reykjavík University

Ermir Pellumbi

Wojciech Woźniak

Yann Le Lorier

[TOC]



## Tasks

### Task 1 - Definition of the Model

#### The state vector $\boldsymbol{x_k}$ 

$\boldsymbol{x_k}$ can be defined as a vector that has as entries $\hat{x}$, $\hat{y}$ the coordinates for the airplane, and $\dot{x}$, $\dot{y}$, the velocity components:
$$
\boldsymbol{\hat{x}_k} = \begin{bmatrix}
\hat{x}\\
\hat{y}\\
\dot{x}\\
\dot{y}
\end{bmatrix}
$$


#### State Transition Matrix $F$

The state transition matrix $F$ can be defined as:

Reference: [State Extrapolation](https://www.kalmanfilter.net/stateextrap.html)
$$
F = 
\begin{pmatrix}
1 & 0 & \Delta t & 0\\
0 & 1 & 0 & \Delta t\\
0 & 0 & 1 & 0\\
0 & 0 & 0 & 1
\end{pmatrix}
$$
In this case, $\Delta t = 10s$ 

For the sake of being thorough, the control input input model $B_k$ is not necessary, since we are assuming  a constant velocity, and the control input $\boldsymbol{u_k} = \begin{bmatrix} 0\\0\\0 \end{bmatrix} $therefore:
$$
B_ku_k= 0
$$


#### Covariance Matrix $Q$ (check)

Reference: [Covariance (Kalman FIlter Tutorial)](https://www.kalmanfilter.net/covextrap.html)

The covariance matrix $Q$ can be drawn from the discrete noise model:


$$
\begin{align}
Q &= 
\begin{pmatrix}
V(x) & 0 & COV(x,\dot{x}) & 0\\
0 & V(y) & 0 & COV(y, \dot{y})\\
COV(\dot{x},x) & 0 & V(\dot{x}) & 0\\
0 & COV(\dot{y},y) & 0 & V(\dot{y})
\end{pmatrix}\\
&= \sigma_a^2
\begin{pmatrix}
\frac{\Delta t^4}{4} & 0 & \frac{\Delta t^3}{2} & 0\\
0 & \frac{\Delta t^4}{4} & 0 & \frac{\Delta t^3}{2}\\
\frac{\Delta t^3}{2} & 0 & \Delta t^2 & 0\\
0 & \frac{\Delta t^3}{2} & 0 & \Delta t^2
\end{pmatrix}
\end{align}
$$


Where $\Delta t = 10s$

#### Observation vector $\boldsymbol{z_k}$

The observation vector $z_k$ can be expressed as follows:
$$
\boldsymbol{z_k} =
\begin{bmatrix}
x\\
y
\end{bmatrix}
$$
#### Observation Matrix $H$

The dimensions of this matrix depend of the dimensions of $x_k$ and $z_k$, we have
$$
\boldsymbol{x_k}=
\begin{bmatrix}
x\\
y\\
\dot{x}\\
\dot{y}
\end{bmatrix}\\
\boldsymbol{z_k} = 
\begin{bmatrix}
z_1\\
z_2
\end{bmatrix}
$$
This way, we know that $H$ has to have a $n_z\times n_x$ dimension:
$$
Dim(H) = 2\times 4
$$
And the following matrix would be the result of $H$:
$$
H=
\begin{bmatrix}
1 & 0 & 0 & 0\\
0 & 1 & 0 & 0
\end{bmatrix}
$$
Let's verify, this way, $\boldsymbol{z_n} - Hx_{n,n}$ yields
$$
\begin{align}
	\boldsymbol{z_n} - Hx_{n,n} &=
	\begin{bmatrix}
		z_1\\
		z_2
	\end{bmatrix}
	-
	\begin{bmatrix}
		1 & 0 & 0 & 0\\
		0 & 1 & 0 & 0
	\end{bmatrix}
	\begin{bmatrix}
		\hat{x}\\
		\hat{y}\\
		\dot{x}\\
		\dot{y}
	\end{bmatrix}\\
	&=
	\begin{bmatrix}
	(z_1-\hat{x})\\
	(z_2-\hat{y})
	\end{bmatrix}
\end{align}
$$


Reference: [State Update (Kalman Filter tutorial)](https://www.kalmanfilter.net/stateUpdate.html)

#### Covariance matrix $R$

The covariance matrix $R$ can be defined as a $2\times 2$ diagonal matrix:
$$
R = 
\begin{pmatrix}
1 & 0\\
0 & 1
\end{pmatrix}
$$

### Task 2 - Understanding the `traffic` data structure

According to the documentation of [Traffic core structure](https://traffic-viz.github.io/core_structure.html), the `traffic` library has three core classes:

- traffic.core.Flight built on top of Pandas Dataframe with attributes:
  - icao24
  - callsign
  - timestamp
  - latitude
  - altitude
- traffic.core.Traffic It's the class that represents a collection of multiple flights, flattened into a single pandas dataframe
- traffic.core.Airspace with properties:
  - area: area of the shape in square meters
  - bounds: returns the bounding box of the shape (west, south, east, north)
  - centroid: returns the centroid of the shape with a `shapely` point
  - extent: returns the extent of the bounding box of the shape (west, east, south, north)

**Plotting some examples from The Quickstart** 

The script for this can be found [here](./KalmanF/test_plots.py)

and can be run with the `IPython` Environment:

```sh
python -m IPython
```

Once inside the `IPython` environment, in the `KalmanF` directory:

```python
run test_plots.py
```



![Plot example](./fig/Plots.png)

### Task 3 - Simulated Radar Measurements

### Task 4,5,6 - Implementation of Kalman Filter Model

see code [here](./KarlmanF/project2.py)

### Task 7 - Errors

We initialized a set of variables that allowed us to see just how much we were improving:

```python
tot_maxi_mean_noised = 0
tot_mean_mean_noised = 0

tot_maxi_mean_filtered = 0
tot_mean_mean_filtered = 0
```

These variables are the means of the noisy flight measurements and the means of the Kalman Filter calculations:

- For the maximum distance mean
- For the mean of the mean of each flight

According to the following tests with all 57 flights:

```
Results:

--------
Unfiltered
        > Maximum distance mean = 346.44509037907585
        > Mean Distance Mean = 80.04527248347601
--------
Filtered
        > Maximum distance mean = 340.1719749305857
        > Mean distance mean = 79.88746612511179
```

this corresponds to a 1.81% improvement in performance for the Maximum distance mean

And a 0.19% improvement in performance for the Mean Distance mean

***

When running standalone tests we realized that:

We can see that the flights where the model performs worse are the flights where there almost no turns, and the Kalman filter lags behind the actual trajectory of the plane, in contrast to flights with lots of turns where the Kalman filter shines.

Another case where this model lacks is when the flight is so long that the curvature of the Earth starts to interfere with the $x$ and $y$ coordinates.

Examples:

**Texas**

![Texas flight](./fig/texas.png)

```
######################### FLIGHT texas ################
----------------------------------------------------
Unfiltered:
        > Maxi Noised-true distance = 377.3392193526951 metres 
        > Noised-true distance mean = 82.63181743435727 metres
----------------------------------------------------
Filtered
        > Maxi Filtered-true distance = 344.78274611002666 metres
        > Filtered-true distance mean = 80.86731657576442 metres
----------------------------------------------------
        > MSE for noised and filtered data = 0.597703142720048
```

**Mecsek_Mountains**

![mecsek](./fig/mecsek.png)

```
######################### FLIGHT mecsek_mountains ################
----------------------------------------------------
Unfiltered:
        > Maxi Noised-true distance = 361.2705218059704 metres 
        > Noised-true distance mean = 79.9102395910746 metres
----------------------------------------------------
Filtered
        > Maxi Filtered-true distance = 317.6362849719422 metres
        > Filtered-true distance mean = 79.47930703683392 metres
----------------------------------------------------
        > MSE for noised and filtered data = 0.6292834555289681
```



### Task 8 - Process noises ($\sigma_p, \sigma_o$)

We conducted the following experiments on the same flight ('Indiana'):

The default configuration is $\sigma_p =1.5$ and $\sigma_o = 50$

Which gives the following results:

```
sigma_p =  1.5
sigma_o =  50
FLIGHT indiana
----------------------------------------------------
Unfiltered:
	> Maxi Noised-true distance = 376.82793983004245 metres 
	> Noised-true distance mean = 80.56738272509013 metres
----------------------------------------------------
Filtered
	> Maxi Filtered-true distance = 335.75696066019503 metres
	> Filtered-true distance mean = 78.7284937840794 metres
----------------------------------------------------
	> MSE for noised and filtered data = 0.6166429105655521
```

***

**Increasing $\boldsymbol{\sigma_p}$**

$\sigma_p = 2.0$ and $\sigma_o = 50$

```
sigma_p =  2.0
sigma_o =  50
FLIGHT indiana
----------------------------------------------------
Unfiltered:
	> Maxi Noised-true distance = 338.20584164354534 metres 
	> Noised-true distance mean = 80.80297335804069 metres
----------------------------------------------------
Filtered
	> Maxi Filtered-true distance = 338.72554024965797 metres
	> Filtered-true distance mean = 78.78773494180938 metres
----------------------------------------------------
	> MSE for noised and filtered data = 0.6513884407566987
```

$\sigma_p=6.0$ and $\sigma_o = 50$

```
sigma_p =  6.0
sigma_o =  50
FLIGHT indiana
----------------------------------------------------
Unfiltered:
	> Maxi Noised-true distance = 311.99729382364444 metres 
	> Noised-true distance mean = 80.4167053596705 metres
----------------------------------------------------
Filtered
	> Maxi Filtered-true distance = 310.6907469168941 metres
	> Filtered-true distance mean = 79.92384829128612 metres
----------------------------------------------------
	> MSE for noised and filtered data = 0.6291971794758533
```

**Increasing  $\boldsymbol{\sigma_o}$**

```
sigma_p =  1.5
sigma_o =  90
FLIGHT indiana
----------------------------------------------------
Unfiltered:
	> Maxi Noised-true distance = 352.4406998448271 metres 
	> Noised-true distance mean = 79.21159442589702 metres
----------------------------------------------------
Filtered
	> Maxi Filtered-true distance = 331.21397474517596 metres
	> Filtered-true distance mean = 82.57125345394844 metres
----------------------------------------------------
	> MSE for noised and filtered data = 0.46013448937142043
```

**Decreasing $\boldsymbol{\sigma_p}$**

```
FLIGHT indiana
----------------------------------------------------
Unfiltered:
	> Maxi Noised-true distance = 351.4009912834685 metres 
	> Noised-true distance mean = 79.75724551207101 metres
----------------------------------------------------
Filtered
	> Maxi Filtered-true distance = 476.3773855865731 metres
	> Filtered-true distance mean = 97.87133152892245 metres
----------------------------------------------------
	> MSE for noised and filtered data = -0.14840329772963845
```

**Decreasing $\boldsymbol{\sigma_o}$**

```
sigma_p =  1.5
sigma_o =  25
FLIGHT indiana
----------------------------------------------------
Unfiltered:
	> Maxi Noised-true distance = 335.3872146187951 metres 
	> Noised-true distance mean = 80.27833389461362 metres
----------------------------------------------------
Filtered
	> Maxi Filtered-true distance = 323.24886105648085 metres
	> Filtered-true distance mean = 78.71728188981585 metres
----------------------------------------------------
	> MSE for noised and filtered data = 0.6460506784266561
```



After all of these observations, we have estimated that the best configuration for $\sigma_p$ and $\sigma_o$ is to have a little bit higher $\sigma_p$, since most of the observations are off by ~80m, so to compensate, the standard deviation from observations has to be around that value.
$$
\sigma_p = 80\\
\sigma_o = 1.5
$$


### Task 9 - Smoothing

After applying smoothing to a few flights we got the following results:

Texas:

![texas](./fig/texas.png)

```
FLIGHT texas:
----------------------------------------------------
Unfiltered:
	> Maxi Noised-true distance = 348.0096414277998 metres 
	> Noised-true distance mean = 79.78908051994826 metres
----------------------------------------------------
Filtered
	> Maxi Filtered-true distance = 343.7114645367926 metres
	> Filtered-true distance mean = 81.21988065270278 metres
----------------------------------------------------
Smoothed
	> Maxi smooth-true distance = 381.2439369164914 metres
	> smoothed-true distance mean = 72.56639291860719 metres
----------------------------------------------------
	> MSE for noised and filtered data = 0.47826039652742763
```



Munich

![munich](./fig/munich.png)



```
FLIGHT munich:
----------------------------------------------------
Unfiltered:
	> Maxi Noised-true distance = 384.4273100913222 metres 
	> Noised-true distance mean = 82.3261516127999 metres
----------------------------------------------------
Filtered
	> Maxi Filtered-true distance = 322.6340576967334 metres
	> Filtered-true distance mean = 84.09167021786106 metres
----------------------------------------------------
Smoothed
	> Maxi smooth-true distance = 298.0388787434447 metres
	> smoothed-true distance mean = 77.93099438515232 metres
----------------------------------------------------
	> MSE for noised and filtered data = 0.5355278359118376
```

Pixair Toulouse

![toulouse](./fig/toulouse.png)

```
FLIGHT pixair_toulouse:
----------------------------------------------------
Unfiltered:
	> Maxi Noised-true distance = 379.84789518501003 metres 
	> Noised-true distance mean = 79.4546558859413 metres
----------------------------------------------------
Filtered
	> Maxi Filtered-true distance = 351.99340681784867 metres
	> Filtered-true distance mean = 83.82174216882619 metres
----------------------------------------------------
Smoothed
	> Maxi smooth-true distance = 319.4323489940619 metres
	> smoothed-true distance mean = 88.78345695131203 metres
----------------------------------------------------
	> MSE for noised and filtered data = 0.4634306102974502
```

We can definitely start to see a pattern. It seems that the smoothing method is preferred over the Kalman Filter when the flights are a lot more complicated. By complicated we mean that there are a lot of turns and not many parts where the flight is going straight.

However, when the flight is simple, like the case of Texas, where there are a lot of parts where the plane is flying straight, the smoothing method seems to have lots of errors.

In conclusion, Smoothing is best when the flight pattern is complex.

### Task 10 - Bonus (3D)

### Setting up the environment

`conda create --prefix kalman-env cartopy shapely python=3.7`

`conda activate ./kalman-env`

**Install traffic**

Has to be made via pip

`pip install traffic`

**Install geopy **

`pip install geopy`

**Install Kalman**

`pip install pykalman`

**Running the code**

We ran all of the tests and the code on the IPython environment (run the following command to avoid using the global python environment):

`python -m IPython`

and once inside the IPython environment:

`run project2.py [flight_name]` e.g. `indiana` or `munich`

