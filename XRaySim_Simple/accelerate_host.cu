#include "accelerate_host.cuh"
#include "accelerate_kern.cu"
#include <stdio.h>
#include <Windows.h>
#include <cutil.h>
#include <cuda_runtime_api.h>
#include "cuda_gl_interop.h"

extern GLuint g_XRayBufferID;
extern GLuint g_posVBO;

rasterData *dev_mRasterData;
float* dev_detectorValues;

bool isFirstRaster = true;
float3 *dev_mVertexList;
uint3 *dev_mFaceList;

bool isFirstBuffer = true;

unsigned char* dev_buffer;

bool isFirstDec = true;

transform* dev_trans;
transform *trans;




extern "C"{ 
	void initRasterData_CUDA(rasterData *mRasterData,
							 float3* vertexList,
							 uint3* faceList,
							 int vertexCount,
							 int faceCount,
							 transform *trans){
 
		if(isFirstRaster)
		{
			cudaMalloc((void**)&dev_mRasterData, sizeof(rasterData)*(faceCount));
			cudaMalloc((void**)&(dev_mVertexList),sizeof(float3)*(vertexCount));
			cudaMalloc((void**)&(dev_mFaceList),sizeof(uint3)*(faceCount));
			cudaMalloc((void**)&(dev_trans),sizeof(transform));
			isFirstRaster = false;

			cudaMemcpy(dev_mFaceList,faceList,sizeof(uint3)*(faceCount),cudaMemcpyHostToDevice);
			cudaMemcpy(dev_mVertexList,vertexList,sizeof(float3)*(vertexCount),cudaMemcpyHostToDevice);

		}
		
		cudaMemcpy(dev_trans,trans,sizeof(transform),cudaMemcpyHostToDevice);

		initRasterData<<<32,faceCount/32+1>>>(dev_mRasterData,dev_mVertexList,dev_mFaceList,faceCount,dev_trans);
		
	}

	void genPicBuffer_CUDA()
	{
 
		int imageSize =   512; 
		
		if(isFirstBuffer) 
		{ 
			isFirstBuffer = false;
		}  
  
		dim3 grids(imageSize/32,imageSize/32);
		dim3 threads(32,32); 
   
		unsigned int bufID = g_XRayBufferID; 
  
		cudaGLMapBufferObject((void**)&dev_buffer,bufID);
		genPicBuffer<<<grids,threads>>>(dev_detectorValues,dev_buffer);
   		cudaGLUnmapBufferObject(bufID); 
   
	} 
	void genDetValues_CUDA(float sourceY,float detY,float increment,float lineSy,int totaFaceCount,
		float *detectorValues)
	{ 
 
		int imageSize = 512;
 
		if(isFirstDec)
		{
			cudaMalloc((void**)&dev_detectorValues, sizeof(float)*imageSize*imageSize);
			isFirstDec = false;
		} 
		cudaMemset(dev_detectorValues,0,sizeof(float)*imageSize*imageSize);

		genDetValues<<<32,totaFaceCount/32+1>>>(dev_mRasterData,sourceY,detY,increment,lineSy,totaFaceCount,
			dev_detectorValues);

	}

	void copyHostMemToDevice(unsigned char* hostBuf)
	{
		unsigned int bufID = g_XRayBufferID; 

		cudaGLMapBufferObject((void**)&dev_buffer,bufID);

		int imageSize = 512;
		cudaMemcpy(dev_buffer,hostBuf,sizeof(unsigned char)*imageSize*imageSize*4,cudaMemcpyHostToDevice);
		cudaGLUnmapBufferObject(bufID); 

	}

 }


