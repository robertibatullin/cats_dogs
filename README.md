# cats_dogs

This is a cat/dog image classifier.

1) Clone this repository:

    git clone https://github.com/robertibatullin/cats_dogs
    
2) Download the model and sample images from Google Drive:

    https://drive.google.com/file/d/1jAvLc9TNRGWYTjh1tYzjiinN32M0N_hA/view?usp=sharing

    Unzip them into cats_dogs folder:
    
    ```
    cats_dogs
            |---model
            |       |--model.hd5
            |---sample
            |---src
    ```

3) To classify all images in a directory, run:

    ```
    python3 predict.py model/model.hd5 <DIRECTORY> [-t <CONFIDENCE THRESHOLD>]
    ```

4) To test images in a directory containing "cats", "dogs" and optionally other subdirectories with already classified images:

    ```
    python3 test.py model/model.hd5 <DIRECTORY> [-t <CONFIDENCE THRESHOLD>] [--n_images <NUMBER OF IMAGES TO TEST IN EACH DIRECTORY>]
    ```
   
5) To retrain the model:

    ```
    python3 train.py model/model.hd5 <TRAIN IMAGES DIRECTORY> <VALIDATION IMAGES DIRECTORY> <NUMBER OF EPOCHS>
    ```

6) To run the Flask app with REST API:

    ```
    python3 rest_api.py
    ```
    
    and send a request like:
    
    ```
    curl -X POST --data-binary @"<IMAGE FILE PATH>" localhost:5000/catdog
    ```

 
