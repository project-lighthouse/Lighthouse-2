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

# Image matching

To extract and save features from the image set you can use the following command:
```bash

$ python ./src/matching/extract_features.py -i ./samples/products-front-back -o ./features.json [--detector={orb, akaze}] [--orb-n-features=2000] [--verbose]

```

To run image matching you can use the following command:
```bash

$ python ./src/matching/match.py -t ./samples/products-front-back/product-1-front.jpg -i ./samples/products-front-back [--detector={orb, akaze}] [--orb-n-features=2000] [--ratio-test-k=0.75] [--n-matches=3] [--no-ui] [--verbose]

```

or if you already have a file with serialized image features:
```bash

$ python ./src/matching/match.py -t ./samples/products-front-back/product-1-front.jpg -d ./features.json [--detector={orb, akaze}] [--orb-n-features=2000] [--ratio-test-k=0.75] [--n-matches=3] [--no-ui] [--verbose]

```

or run `$ python ./src/matching/match.py -h` to see all available options.