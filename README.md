# Quora Insincere Questions Classification

[![CircleCI](https://circleci.com/gh/k-fujikawa/qiqc.svg?style=svg&circle-token=5016f93e46d89ad825834ac2478f1cce8b4f407b)](https://circleci.com/gh/k-fujikawa/qiqc)

This is a repository for kaggle competition: [Quora Insincere Questions Classification](https://www.kaggle.com/c/quora-insincere-questions-classification)

## Requirement

- docker >= 17.12.0
- docker-compose >= 1.19.0
- nvidia-docker2 >= 2.0.2

## Getting started

### :beginner: Setup

#### Setup kaggle API credentials

Download kaggle.json and place in the location: `~/.kaggle/kaggle.json`.  
See details: https://github.com/Kaggle/kaggle-api

#### Build Docker image

```
docker-compose build
```

#### Download and unzip competition datasets

```
docker-compose run cpu kaggle competitions download quora-insincere-questions-classification -p input
unzip "input/*.zip" -d input
```

### :rocket: Execute experiments

```
docker-compose run gpu python exec/train-cv.py -m <path-to-models>
```
