#ifndef   ACCELERATE_HOST_H_
#define   ACCELERATE_HOST_H_

#include "common.h"
#include "vector_types.h"
extern "C"
{

	void initRasterData_CUDA(rasterData *mRasterData,
							 float3* vertexList,
							 uint3* faceList,
							 int vertexCount,
							 int faceCount,
							 transform* trans);

	void genPicBuffer_CUDA();

	void genDetValues_CUDA(float sourceY,float detY,float increment,float lineSy,int totaFaceCount,
		float *detectorValues);
	
	void copyHostMemToDevice(unsigned char* hostBuf);
}
#endif