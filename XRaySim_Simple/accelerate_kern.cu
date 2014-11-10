	


#include <math.h>
#include "sm_12_atomic_functions.h"
#include "common.h"

__global__ void transVertexList(int vertexCount,float3 *mVertexList, transform* trans)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if(tid < vertexCount)
	{
		float x,y,z;

		x = mVertexList[tid].x;
		y = mVertexList[tid].y;
		z = mVertexList[tid].z;

		mVertexList[tid].x = x*cos(trans->zRot)*cos(trans->yRot)+y*(-sin(trans->zRot)*cos(trans->xRot)*cos(trans->yRot)+sin(trans->xRot)*sin(trans->yRot))+z*(sin(trans->zRot)*sin(trans->xRot)*cos(trans->yRot)+sin(trans->yRot)*cos(trans->xRot))+trans->xTrans;
		mVertexList[tid].y = x*sin(trans->zRot)+y*cos(trans->zRot)*cos(trans->xRot)-z*cos(trans->zRot)*sin(trans->xRot);
		mVertexList[tid].z = -x*sin(trans->yRot)*cos(trans->zRot)+y*(sin(trans->zRot)*cos(trans->xRot)*sin(trans->yRot)+sin(trans->xRot)*cos(trans->yRot))+z*(-sin(trans->zRot)*sin(trans->xRot)*sin(trans->yRot)+cos(trans->yRot)*cos(trans->xRot))+trans->zTrans;

	}
	__syncthreads();
}

__global__ void initRasterData(rasterData *mRasterData,
							   float3 *mVertexList,
							   uint3 *mFaceList,
							   int faceCount,
							   transform* trans){

		int tid = threadIdx.x + blockIdx.x * blockDim.x;

		if( tid < faceCount )
		{
			float x,y,z;

			x = mVertexList[mFaceList[tid].x].x;
			y = mVertexList[mFaceList[tid].x].y;
			z = mVertexList[mFaceList[tid].x].z;

			mRasterData[tid].v1x = x*cos(trans->zRot)*cos(trans->yRot)+y*(-sin(trans->zRot)*cos(trans->xRot)*cos(trans->yRot)+sin(trans->xRot)*sin(trans->yRot))+z*(sin(trans->zRot)*sin(trans->xRot)*cos(trans->yRot)+sin(trans->yRot)*cos(trans->xRot))+trans->xTrans;
			mRasterData[tid].v1y = x*sin(trans->zRot)+y*cos(trans->zRot)*cos(trans->xRot)-z*cos(trans->zRot)*sin(trans->xRot);
			mRasterData[tid].v1z = -x*sin(trans->yRot)*cos(trans->zRot)+y*(sin(trans->zRot)*cos(trans->xRot)*sin(trans->yRot)+sin(trans->xRot)*cos(trans->yRot))+z*(-sin(trans->zRot)*sin(trans->xRot)*sin(trans->yRot)+cos(trans->yRot)*cos(trans->xRot))+trans->zTrans;

			x = mVertexList[mFaceList[tid].y].x;
			y = mVertexList[mFaceList[tid].y].y;
			z = mVertexList[mFaceList[tid].y].z;

			mRasterData[tid].v2x = x*cos(trans->zRot)*cos(trans->yRot)+y*(-sin(trans->zRot)*cos(trans->xRot)*cos(trans->yRot)+sin(trans->xRot)*sin(trans->yRot))+z*(sin(trans->zRot)*sin(trans->xRot)*cos(trans->yRot)+sin(trans->yRot)*cos(trans->xRot))+trans->xTrans;
			mRasterData[tid].v2y = x*sin(trans->zRot)+y*cos(trans->zRot)*cos(trans->xRot)-z*cos(trans->zRot)*sin(trans->xRot);
			mRasterData[tid].v2z = -x*sin(trans->yRot)*cos(trans->zRot)+y*(sin(trans->zRot)*cos(trans->xRot)*sin(trans->yRot)+sin(trans->xRot)*cos(trans->yRot))+z*(-sin(trans->zRot)*sin(trans->xRot)*sin(trans->yRot)+cos(trans->yRot)*cos(trans->xRot))+trans->zTrans;

			x = mVertexList[mFaceList[tid].z].x;
			y = mVertexList[mFaceList[tid].z].y;
			z = mVertexList[mFaceList[tid].z].z;

			mRasterData[tid].v3x = x*cos(trans->zRot)*cos(trans->yRot)+y*(-sin(trans->zRot)*cos(trans->xRot)*cos(trans->yRot)+sin(trans->xRot)*sin(trans->yRot))+z*(sin(trans->zRot)*sin(trans->xRot)*cos(trans->yRot)+sin(trans->yRot)*cos(trans->xRot))+trans->xTrans;
			mRasterData[tid].v3y = x*sin(trans->zRot)+y*cos(trans->zRot)*cos(trans->xRot)-z*cos(trans->zRot)*sin(trans->xRot);
			mRasterData[tid].v3z = -x*sin(trans->yRot)*cos(trans->zRot)+y*(sin(trans->zRot)*cos(trans->xRot)*sin(trans->yRot)+sin(trans->xRot)*cos(trans->yRot))+z*(-sin(trans->zRot)*sin(trans->xRot)*sin(trans->yRot)+cos(trans->yRot)*cos(trans->xRot))+trans->zTrans;

			mRasterData[tid].e1x = mRasterData[tid].v2x - mRasterData[tid].v1x;
			mRasterData[tid].e1y = mRasterData[tid].v2y - mRasterData[tid].v1y;
			mRasterData[tid].e1z = mRasterData[tid].v2z - mRasterData[tid].v1z;
			mRasterData[tid].e2x = mRasterData[tid].v3x - mRasterData[tid].v1x;
			mRasterData[tid].e2y = mRasterData[tid].v3y - mRasterData[tid].v1y;
			mRasterData[tid].e2z = mRasterData[tid].v3z - mRasterData[tid].v1z;

			mRasterData[tid].nx = mRasterData[tid].e1y*mRasterData[tid].e2z - mRasterData[tid].e1z*mRasterData[tid].e2y;
			mRasterData[tid].ny = mRasterData[tid].e1z*mRasterData[tid].e2x - mRasterData[tid].e1x*mRasterData[tid].e2z;
			mRasterData[tid].nz = mRasterData[tid].e1x*mRasterData[tid].e2y - mRasterData[tid].e1y*mRasterData[tid].e2x;

			double tempMag = sqrtf(mRasterData[tid].nx * mRasterData[tid].nx
				+	mRasterData[tid].ny * mRasterData[tid].ny
				+	mRasterData[tid].nz * mRasterData[tid].nz);

			if (tempMag > 0)
			{
				mRasterData[tid].nx /= tempMag;
				mRasterData[tid].ny /= tempMag;
				mRasterData[tid].nz /= tempMag;
			}

		}
		__syncthreads();
	}

__global__ void genPicBuffer(float* detectorValues,
							 unsigned char* buffer
							 )
{
	int j = threadIdx.x + blockIdx.x * blockDim.x;
	int i = threadIdx.y + blockIdx.y * blockDim.y;

	int imageSize = 512;

	int ceilTemp=0;

	ceilTemp = ceil(300.0*detectorValues[i*imageSize + j]);

	buffer[(i + imageSize*j)*4+0] = ceilTemp;
	buffer[(i + imageSize*j)*4+1] = ceilTemp;
	buffer[(i + imageSize*j)*4+2] = ceilTemp;
	buffer[(i + imageSize*j)*4+3] = ceilTemp;

	__syncthreads();
}

__global__ void genDetValues(rasterData *mRasterData,float srcY,float dY,float inct,
							 float liSy,int totaFaceCount,float *detectorValues)
{
	float dir[3] = {0};
	float tempMag = 0;
	float rayTemp=0;
	
	float increment = inct;
	float sourceX = 0;
	float sourceY = srcY;
	float sourceZ = 0;
	float vert1[2] = {0},vert2[2]={0}, vert3[2]={0};
	float minX, maxX, minY, maxY;
	float e1[2], e2[2], e3[2];
	float lineSx, lineSy, lineSz;
	float lineDx, lineDy, lineDz;
	float temp, temp4, temp5, temp6;
	float testX, testY,u,v,wb;
    int maxXi, minXi, maxYi, minYi;
	int imageSize=512;
	int halfImage=256;
	maxX=-1000.0,maxY=-1000.0;
	minX=1000.0,minY=1000.0;
	maxXi = minXi = maxYi = minYi =0;
	u=0,v=0,wb=0;
	lineSx=0;
	lineSy=liSy;
	lineSz=0;
	lineDx=0;
	lineDy=1.0;
	lineDz=0;
	float detX,detY,detZ;
	detX=detZ=0;
	detY=dY;
	//
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid < totaFaceCount)
	{
		dir[0] = mRasterData[tid].v1x - sourceX;
		dir[1] = mRasterData[tid].v1y - sourceY;
		dir[2] = mRasterData[tid].v1z - sourceZ;
		tempMag = sqrtf(dir[0]*dir[0]+dir[1]*dir[1]+dir[2]*dir[2]);

		/*if(tempMag > 0)*/
		{
			dir[0]/=tempMag;
			dir[1]/=tempMag;
			dir[2]/=tempMag;
		}

		rayTemp = (detY - sourceY)/(dir[1]);

		vert1[0] = sourceX + (rayTemp*dir[0]);
		vert1[1] = sourceZ + (rayTemp*dir[2]);

		if(vert1[0]<minX)
			minX=vert1[0];
		if(vert1[0]>maxX)
			maxX=vert1[0];
		if(vert1[1]<minY)
			minY=vert1[1];
		if(vert1[1]>maxY)
			maxY=vert1[1];

		dir[0] = mRasterData[tid].v2x - sourceX;
		dir[1] = mRasterData[tid].v2y - sourceY;
		dir[2] = mRasterData[tid].v2z - sourceZ;
		tempMag = sqrtf(dir[0]*dir[0]+dir[1]*dir[1]+dir[2]*dir[2]);

		/*if(tempMag > 0)*/
		{
			dir[0]/=tempMag;
			dir[1]/=tempMag;
			dir[2]/=tempMag;
		}

		rayTemp = (detY - sourceY)/(dir[1]);

		vert2[0] = sourceX + (rayTemp*dir[0]);
		vert2[1] = sourceZ + (rayTemp*dir[2]);

		if(vert2[0]<minX)
			minX=vert2[0];
		if(vert2[0]>maxX)
			maxX=vert2[0];
		if(vert2[1]<minY)
			minY=vert2[1];
		if(vert2[1]>maxY)
			maxY=vert2[1];

		dir[0] = mRasterData[tid].v3x - sourceX;
		dir[1] = mRasterData[tid].v3y - sourceY;
		dir[2] = mRasterData[tid].v3z - sourceZ;
		tempMag = sqrtf(dir[0]*dir[0]+dir[1]*dir[1]+dir[2]*dir[2]);

		/*if(tempMag > 0)*/
		{
			dir[0]/=tempMag;
			dir[1]/=tempMag;
			dir[2]/=tempMag;
		}

		rayTemp = (detY - sourceY)/(dir[1]);

		vert3[0] = sourceX + (rayTemp*dir[0]);
		vert3[1] = sourceZ + (rayTemp*dir[2]);


		if(vert3[0]<minX)
			minX=vert3[0];
		if(vert3[0]>maxX)
			maxX=vert3[0];
		if(vert3[1]<minY)
			minY=vert3[1];
		if(vert3[1]>maxY)
			maxY=vert3[1];

		minX/=increment;
		maxX/=increment;
		minY/=increment;
		maxY/=increment;

		maxXi = (int)maxX;
		minXi = (int)minX;
		maxYi = (int)maxY;
		minYi = (int)minY;

		e2[0] = vert2[0] - vert1[0];
		e2[1] = vert2[1] - vert1[1];
		e3[0] = vert3[0] - vert1[0];
		e3[1] = vert3[1] - vert1[1];

		lineSx = increment*((maxX+minX)/2);
		lineSz = increment*((maxY+minY)/2);

		lineDx = lineSx - sourceX;
		lineDy = lineSy - sourceY;
		lineDz = lineSz - sourceZ;

		temp5 = sqrtf(lineDx*lineDx + lineDy*lineDy + lineDz*lineDz);

		/*if(temp5 > 0)*/
		{
			lineDx/=temp5;
			lineDy/=temp5;
			lineDz/=temp5;
		}

		temp4 = lineDx*mRasterData[tid].nx + lineDy*mRasterData[tid].ny + lineDz*mRasterData[tid].nz;

		if(temp4<0)
			temp6 = -1.0;
		else
			temp6 = 1.0;

		register float v1y,v2y,v3y;
		v1y = mRasterData[tid].v1y;
		v2y = mRasterData[tid].v2y;
		v3y = mRasterData[tid].v3y;

		register float ke[2][2] = {0};
		float temp3=0;

		temp3 = (e3[0]*e2[1]) - (e3[1]*e2[0]);
		/*if(temp3 !=0)*/
			temp3 = 1/temp3;
		testX = (minXi)*increment - vert1[0];
		testY = (minYi)*increment - vert1[1];

		ke[0][0] = temp3*e2[0];
		ke[0][1] = temp3*e2[1];
		ke[1][0] = temp3*e3[0];
		ke[1][1] = temp3*e3[1];

		if(minXi>maxXi)
			return;
		if(minYi>maxYi)
			return;
		if(minXi<-halfImage)
			minXi = -halfImage;
		if(minYi<-halfImage)
			minYi = -halfImage;
		if(maxXi>halfImage)
			maxXi = halfImage;
		if(maxYi>halfImage)
			maxYi = halfImage;


		for(int x=minXi; x<=maxXi; x++)
		{
			for(int y=minYi;y<=maxYi;y++)
			{
				if(x<-halfImage || x>halfImage || y<-halfImage || y>halfImage)
					continue;
				//Compute barycentric co-ords (BCC) of x,y (integers)
				//If valid, add BCC*z determine new z depth and fill pixel

				u = ke[1][0]*testY - ke[1][1]*testX;
				v = ke[0][1]*testX - ke[0][0]*testY;
				wb = 1.0-u-v;

				testY += increment;

				if(u>1.0 || v>1.0 || wb>1.0 || u<0 || v<0 || wb<0)
					continue;

				//Now we have BCCs u and v, determine depth and add
				temp = sourceY - (wb*v1y) - (u*v2y) - (v*v3y);

				atomicAdd(&(detectorValues[(halfImage+x)*imageSize + halfImage+y]),temp6*temp);
			}
			testX += increment;
			testY = minYi*increment - vert1[1];
		}
	}
	__syncthreads();
}
