
# Practicing with the DIGITS Workflow

In this project, you will be leveraging NVIDIA’s DIGITS workflow to rapidly prototype ideas that can be deployed on the Jetson in close to real time.With DIGITS, you’ll prototype classification networks, detection networks, segmentation networks and even more!

![image.png](attachment:image.png)

In the first part of this project, you’ll practice with the DIGITS workflow on data that is provided for you.

We will be leveraging the power of Udacity's new workspaces with the power of NVIDIA GPUs. You will be allotted 50 hours of GPU compute time so be vigilant with your usage. There is no limit on instances that are in CPU mode or use CPU mode. The DIGITS workspace can be found at the end of this module.

## Start DIGITS

Start the DIGITS server by entering the command digits into a terminal. This will begin the boot of the DIGITS server (it will take a minute).

Now, from another terminal run print_connection.sh in order to get the link for the DIGITS GUI (Graphical User Interface). Keeping this script running will keep your workspace active if you are training a network but be sure to quit it if you are not using the workspace as it will use all of your GPU hours.

## The Data

Let’s pause for a moment to talk about the data you will be training.

These are photos taken from a Jetson mounted over a conveyor belt.

We are training pictures of candy boxes, bottles, and nothing (empty conveyor belt) for the purpose of real time sorting. This kind of design can be extrapolated to many things that require real time sorting.

Here are some examples of the data:

![image.png](attachment:image.png)

Add the supplied dataset into DIGITS. It can be found in the data directory (/data/P1_data).

The provided data has the following file structure:
P1_data/
├── Bottle/
│   ├── Bottle_1.png
│   └── Bottle_2.png
├── Candy_box/
│   ├── Candy_box_1.png
│   └── Candy_box_2.png
└── Nothing/
    ├── Nothing_1.png
    └── Nothing_2.png
Once your data is imported, it is up to you to choose a training model. You can use a pre-supplied one, one from the DIGITS model store, an external network or even customize the above choices.

Your model will have to achieve an inference time of 10 ms or less on the workspace and have an accuracy greater than 75 percent.

## Evaluate the model

Test your trained model by running the command evaluate in another terminal with the DIGITS server still running, but only after you are done training your model. It will ask you for the model’s job id which can be found here:

![image.png](attachment:image.png)

It will then print out the results of this model. The evaluate command checks the inference speed of your model for a single input averaged over ten attempts for five runs. Take a screenshot of it to include in your write-up. It uses Tensor RT 3.0 to achieve this in a very fast time. Then your model is tested on a test set that is not used for testing or validation. Do not spend too much time trying to make a model work above the required results. It is better to spend that additional energy working on your own Robotic Inference idea, which we will talk about next!

Note: The evaluate command will not work with your custom model.

## Project guidelines:

In addition to training a network on the supplied data, you will also need to choose and train a network using your own collected data. At a minimum, it needs to be a classification network with at least 3 classes. However, if you would like to be more adventurous you can use more than 3 classes and even subclasses!

If you are looking for an extra challenge, you can create a detection network. It will require you to annotate your data in addition to collecting it. More information can be found in the next section on this process.

Its okay to use a sample idea below if you’re having a hard time deciding what to do!

### Resources and ideas:

Pill identifier with classes: (pill a, pill b, pill c, no pill)
Defective item vs normal item with classes: (no item, defective item, normal item)
Person vs no person with classes: (correct person, wrong person, no person)
Location of robot part on a workbench.
Insert your idea here!
Fun examples to check out!

https://github.com/S4WRXTTCS/jetson-inference

## Classification Network

If you want to create a classification network, then at least 400 images are recommended per class but it can vary. This number is very subjective but is a good starting point. It will be up to you to determine the right number of samples for your project. Your network may do well at learning one class but struggle with another and therefore more of that data will need to be collected. Also, remember to collect images in the same environment in which you will be conducting your inference.

## Detection Network

If you decide to work with a detection network, you will need to annotate your data before uploading it to DIGITS. This means that you need to put bounding boxes around what you want your network to learn. There are a lot of different software applications out there that can help expedite this for you. Here are some options for image annotation and there are many more that can be found with the help of an online search. Choose one you think you will be most proficient in.

## Collecting the images

There are a number of ways to collect images. You can use a webcam and a Python or C++ script to collect data, for example. Some people have used phones to collect data. If you have a Jetson, it can be used to collect data as well.

We will provide a basic python script for collecting images from your webcam but you are encouraged to find other methods as well.

If you are limited on upload speed for your large data set, it is advised to upload your data to a Google Drive or other fast cloud storage. Once it’s uploaded, you can download your data into your instance in a very short amount of time.

Example Python Data Capture Script
Note: You have to setup the proper environment with Python 2.7 before running this script using cv2: 
conda install -c conda-forge opencv=2.4

```python
import cv2

# Run this script from the same directory as your Data folder

# Grab your webcam on local machine
cap = cv2.VideoCapture(0)

# Give image a name type
name_type = 'Small_cat'

# Initialize photo count
number = 0

# Specify the name of the directory that has been premade and be sure that it's the name of your class
# Remember this directory name serves as your datas label for that particular class
set_dir = 'Cat'

print ("Photo capture enabled! Press esc to take photos!")

while True:
    # Read in single frame from webcam
    ret, frame = cap.read()

    # Use this line locally to display the current frame
    cv2.imshow('Color Picture', frame)

    # Use esc to take photos when you're ready
    if cv2.waitKey(1) & 0xFF == 27:

        # If you want them gray
        #gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        # If you want to resize the image
        # gray_resize = cv2.resize(gray,(360,360), interpolation = cv2.INTER_NEAREST)

        # Save the image
        cv2.imwrite('Data/' + set_dir + '/' + name_type + "_" + str(number) + ".png", frame)

        print ("Saving image number: " + str(number))

        number+=1

    # Press q to quit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## Optional Deployment

Now that you have trained your network, ready to see it in action? Great - let’s get started!

The first step is to download your model from DIGITS to your Jetson. To do this, navigate in a browser on the Jetson device to your DIGITS server. From there download your model, like so:

![image.png](attachment:image.png)

Next, create a folder on the system with a name for your model and extract the contents of your downloaded file into that folder with the tar -xzvf command.

Then, create an environment variable called NET to the location of the model path. Do this by entering something like export NET=/home/user/Desktop/my_model into the terminal. The exact command will depend on the shell you are using.

Navigate to the Jetson inference folder then into the executable binaries and launch imagenet or detect net like so:

./imagenet-camera --prototxt=$NET/deploy.prototxt --model=$NET/your_model_name.caffemodel --labels=$NET/labels.txt --input_blob=data --output_blob=softmax

You will then observe real time results from your Jetson camera!

If you desire to actuate based on the information the classifier is providing, you can edit either the imagenet or detectnet c++ file accordingly!

Here is an example of calling a servo action based on a classification result. It can be inserted into the code here at line 168:


```python
std::string class_str(net->GetClassDesc(img_class));

if("Bottle" == class_str){
          cout << "Bottle" << endl;
          // Invoke servo action
}

else if("Candy_Box" == class_str){
         cout << "Candy_Box" << endl;
         // Do not invoke servo action
}

else {
         // Catch anything else here
}

```
