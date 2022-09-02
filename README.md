# Advertisement Banner Replacement

This projects shows how to replace advertisement banners with any other template advertisement using Computer Vision and OpenCV in C++.

## Algorithm steps
Step 1: Detect the advertisement banners in the image.
Step 2: Replace the advertisement banner with the template advertisement.

The code can be executed from the build directory with the folling dummy command:
```
./ad-banner-replacement <<your_path_to_input_image/video>> <<your_path_to_ad_template>>
```

The output will then be saved to the same directory as the input image/video, in .jpeg or .mov format respectively.

## Results on images
Example 1, input image:
![banner1](https://user-images.githubusercontent.com/43403875/188171887-8a6325b9-31e3-40d7-b5c6-5c241ecbbaec.jpeg) ![banner1_result](https://user-images.githubusercontent.com/43403875/188171905-fb11870c-810d-4ba0-8ab8-bdc461de159f.jpeg)
Result after the advertisement has been replaced with kinder chocolate bars:

Example 2, input image:
![banner2](https://user-images.githubusercontent.com/43403875/188172130-8fc854b8-4c29-452e-83a0-de496170512e.jpeg)
Result after the advertisement has been replaced with lego bricks:
![banner2_result](https://user-images.githubusercontent.com/43403875/188172138-6308c74c-b70c-4345-a782-23f0f5d34cba.jpeg)

## Reslts on videos:
Result after the the algorithm has been applied on the entire video:
![modric](https://user-images.githubusercontent.com/43403875/188172240-56d098da-b6e5-4422-ade0-681dfa40879a.gif)
