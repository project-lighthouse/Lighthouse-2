# More experiment for helping out blind people

For the moment, these experiments concentrate on detecting objects shaken in
front of the camera. Later, we will try to support a moving camera, as well
as storing the picture, matching new pictures against stored ones.

# Basic usage

Launch:

```sh
python src/main.py
```

Move an object in front of the camera. It will try to isolate what is moving.

# Examples
```
$ python ./src/main.py --matching-orb-n-features 500 --db-store-images --acquisition-keep-objects 5 --matching-score-threshold 10 --no-gui --cmd-ui --verbose
```