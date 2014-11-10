## A simple real-time X-Ray simulator program ##

Tags: `OpenGL` `CUDA` `Qt` `X-Ray` `Real-time`

This is a simple real-time X-Ray simulator program using CUDA for GPUs from NVidia.

The code relies on Visual Studio 2008 for compilation, Qt for user interface, CUDA for parallel computing and OpenGL for visualization.

First of all, Qt and CUDA must be both installed correctly on your computer. Both are recommended to installed to the default path. For example,

    QT_PATH <-- C:\Qt\4.7.0
    CUDA_SDK_PATH <-- C:\ProgramData\NVIDIA Corporation\NVIDIA GPU Computing SDK 4.0
    CUDA_PATH_V4_0 <-- C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v4.0

If not, you should manually change the "include" and "lib" path in the project properties.
If you are using a Qt VS add-in tool, there's nothing you should do with Qt. It should be noticed that 2 ''include'' and 2 ''lib'' paths for CUDA and OpenGL should be manually added.

    $(CUDA_SDK_PATH)\C\common\inc
    $(CUDA_PATH_V4_0)\include
    $(CUDA_SDK_PATH)\C\common\lib\Win32
    $(CUDA_PATH_V4_0)\lib\Win32

However, the project is only tested with Qt 4.7 and CUDA 4.0. There may be some errors when it is compiled with other versions of Qt and CUDA, especially the latter ones.

A snapshot of this program is showed below

![snapshot.jpg](https://bytebucket.org/HsiaTsing/simplexraysimulator/raw/dc444fae3b16c1d40d572304803e6d7beb953c72/snapshot.jpg?token=ea7d27928778a68844ab05bf18b7c51b9a55a48d)

This work is finished when I was in the State Key Lab of Virtual Reality Technology and System, Beihang University.