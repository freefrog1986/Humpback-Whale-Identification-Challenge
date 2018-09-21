This is Udacity Machine Learning Engineer (Advanced) nanodegree finall project.

To predict whale Ids from humpback whale images, we gathered thousands of images of humpback whale flukes with appropriate Id and build a model try to resolve the problem. 

We built a shallow CNN model with 8 layers as our benchmark model and a more complexed model VGG16 to try to get better performance. And 4 preprocessing steps were adopt before building the model.

Finally our model were tested both on training data and testing data with accuracy  as the algorithm metrics, we also uploading the final result to the official competition website to get MAP@5 score.

There are 3 python script named benchmark.py, vgg16_fineturning.py and vgg16.py. Those three scripts including three diifferent model. You can try one of them by running the script. 

The dataset is over 700MB so I didn't uploaded, you can download all datasets in [this page](https://www.kaggle.com/c/whale-categorization-playground/data). 
