# computer-vision-project

## Topic
Our team will be developing an object-specific property swapping filter. This filter would find similar objects between two images and transfer one object’s properties in the first image, like color, brightness, and contrast, to the other object in the second image.

## Description
The filter works uniquely in that it’ll take two images, segment the objects in the images, analyze the properties of those objects, and transfer those properties to the target image. An example of functionality would be having one image that contains a segmented sky and a second image that also contains a sky. The values of each sky object would be extracted, then switched to the other sky, so the first image’s sky would now look more similar in color, brightness, contrast, etc. to the second image’s sky. The same would happen to the second image’s sky, which would then look like the first image’s.
If the input images are lacking in similar objects, then the application should use the scale and position of the objects to determine which objects are similar and need their values swapped. That way, the filter will always give an output. If there are more object classes found in one image than in the other, then the objects that are least similar to anything in the image with fewer objects will not have their property filters applied. 
There should be an option to only return one output image if the user so chooses. In that case, the input and filter image would need to be specified. 

## How to Run

To run the project, run rcnn.py

Resnet - segmenting
unet - Homemade untrained model.