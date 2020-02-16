# Helmut
Nudity detection neural network built with TensorFlow. I used 250 images for the training set, I gave the name in honor of the photographer [Helmut Newton](https://pt.wikipedia.org/wiki/Helmut_Newton)


## How to run:
``` 
python helmut.py [imagepath]
```

Or instead you can run the server:
``` 
python server.py
```
and give a POST request to it
```
curl -F "file=@[imagepath]" http://127.0.0.1:5000/
```

It will return a json like that:
{"notnude": 0.9963597655296326, "nude": 0.003640185110270977}
