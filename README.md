This is a plug-and-play python3 module, which can do hand tracking and joint angle estimation given a RGB video.

# Setup
```
conda create -n wilor python=3.10
conda activate wilor
pip install -r requirements.txt --no-build-isolation
```
It may take few mins.

# Run wilor
```
bash run_wilor.sh
```
The results are in ```./test_data/wilor_output/```

You could check ```./doc``` about the order of hand keypoints and the definition of joint angle 

# One Example
![door opening](https://github.com/user-attachments/assets/52863c42-d025-4550-b4ff-1a36a93d55c3)

Thanks for the help from [WiLor](https://github.com/rolpotamias/WiLoR) and [WiLor-mini](https://github.com/warmshao/WiLoR-mini)