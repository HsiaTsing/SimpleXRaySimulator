#ifndef MODEL_H
#define MODEL_H
#include "vector_types.h"
class Model
{

public:
	
	bool initialized;

	float3 *m_hPoints;
	int m_nPoints;
	
	uint3 *m_hFaces;
	int m_nFaces;

	float3 *m_hNormals;
	int m_nNormals;

	float scaleFactor;

	unsigned int m_VA0;
	unsigned int m_posVBO;
	unsigned int m_normVBO;
	unsigned int m_elemVBO;

	float3 m_min;
	float3 m_max;

	void init();

	void load(const char* filename, float SF, bool recenter, bool recalcNormals);

	void draw();

	void computeBoundingBox(bool recenter);

};

#endif