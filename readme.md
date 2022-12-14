# SELECTIVE FACE BLUR
Have you ever happened to take a picture of yourself or someone else, with unwanted visitors in the background?
Or perhaps you just want to remove someone's face in an old group picture for whatever reason?

This very easy to use program let's you get rid of non-desired faces across any image!

## Dependencies:
* [Python](https://www.python.org/doc/) - 3.10.5
* [OpenCV](https://docs.opencv.org/4.6.0/) - 4.6.0
* [Numpy](https://numpy.org/doc/stable/) - 1.22.4
* [Streamlit](https://docs.streamlit.io/library/get-started) - 1.10.0 (Only required to run the streamlit app version)
## How to use:
To run the Streamlit app locally, just go to *../src/streamlit* inside this workspace and type:

```console
    $ streamlit run ImageFilterApp.py
```

![alt text](https://github.com/Josgonmar/Selective-face-blur/blob/master/visuals/interface.jpg?raw=true)

A new tab will be opened in your favourite web browser where you have to upload a picture.
Once it's uploaded, modify both the face detector threshold and the blur factor to get the desired results.
Finally select which face(s) you'd like to apply the blur, and download the final image!
## License:
Feel free to use this programa whatever you like!
