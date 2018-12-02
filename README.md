# chainer-docker

[![CircleCI](https://circleci.com/gh/k-fujikawa/chainer-docker.svg?style=svg)](https://circleci.com/gh/k-fujikawa/chainer-docker)

chainer-docker is a template for a typical machine learning project using chainer, docker, and docker-compose.

## Requirement

- docker >= 17.12.0
- docker-compose >= 1.19.0
- nvidia-docker2 >= 2.0.2

## Getting start

### :beginner: Setup configuation

Register the required environment variables to your `.bashrc`:

```bash
export UID
export GID=`id -g`
```

Register the required environment variables to `.env`:

- IMAGENAME: Name of docker image for this project (default: chainer-docker)

And set the name of this package in setup.py:

```python
...

setup(
    name='name of package',

...
```

### :whale: Use docker without GPU (ex. Mac OS)

#### Build docker image

```
$ docker-compose build cpu
```

#### Run docker container

You can run a command in a new container:

```
$ docker-compose run cpu [COMMAND]
```

### :whale: Use docker with GPU

#### Build docker image

```
$ docker-compose build gpu
```

#### Run docker container

Check if a GPU is available in the container by running `nvidia-smi`:

```
$ docker-compose run gpu nvidia-smi
```

And then you can run a command in a new container:

```
$ docker-compose run gpu [COMMAND]
```

### :blue_book: Launch Jupyter Lab

Start up Jupyter Lab using:

```
$ docker-compose up jupyter
```

You will get the response as follows:

```
...
Copy/paste this URL into your browser when you connect for the first time,
    to login with a token:
        http://0.0.0.0:8888/?token=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
...
```

Then you can access jupyter lab from the follwing URL:

```
http://0.0.0.0:8888/?token=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

Note that you can change the port using environment variable as below:

```
$ PORT=9999 docker-compose up jupyter
```