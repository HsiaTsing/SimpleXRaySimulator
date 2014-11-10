#ifndef COMMON_H
#define COMMON_H

struct rasterData
{
	float v1x, v1y, v1z;
	float v2x, v2y, v2z;
	float v3x, v3y, v3z;
	float e1x, e1y, e1z;
	float e2x, e2y, e2z;
	float boxX1, boxX2;
	float boxY1, boxY2;
	float nx, ny, nz;
};

struct transform
{
	float xRot;
	float yRot;
	float zRot;
	float xTrans;
	float yTrans;
	float zTrans;
};

#define ENERGY_BAND_WIDTH 1.0
#define MAX_PATH_LENGTH 1.5

#endif